// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// Author: Haoyi Wu

import cpp from 'child-process-promise';
import cp from 'child_process';
import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import ts from 'tail-stream';
import tkill from 'tree-kill';
import { NNIError, NNIErrorNames } from 'common/errors';
import { getExperimentId } from 'common/experimentStartupInfo';
import { getLogger, Logger } from 'common/log';
import { powershellString, shellString, createScriptFile } from 'common/shellUtils';
import {
    HyperParameters, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, TrialJobStatus
} from 'common/trainingService';
import {
    delay, generateParamFileName, getExperimentRootDir, getJobCancelStatus, getNewLine, isAlive, uniqueString
} from 'common/utils';
import { SlurmConfig } from 'common/experimentConfig';
import { execMkdir, execNewFile, getScriptName, setEnvironmentVariable } from '../common/util';
import { Deferred } from 'ts-deferred';

/**
 * Decode a command
 * @param Buffer binary incoming data
 * @returns a tuple of (success, commandType, content, remain)
 *          success: true if the buffer contains at least one complete command; otherwise false
 *          remain: remaining data after the first command
 */
function decodeCommand(data: Buffer): [boolean, string, string, Buffer] {
    if (data.length < 8) {
        return [false, '', '', data];
    }
    const commandType: string = data.slice(0, 2).toString();
    const contentLength: number = parseInt(data.slice(2, 8).toString(), 10);
    if (data.length < contentLength + 8) {
        return [false, '', '', data];
    }
    const content: string = data.slice(8, contentLength + 8).toString();
    const remain: Buffer = data.slice(contentLength + 8);

    return [true, commandType, content, remain];
}

/**
 * SlurmTrialJobDetail
 */
class SlurmTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: TrialJobApplicationForm;
    public slurmJobId?: number;

    constructor(
        id: string, status: TrialJobStatus, submitTime: number,
        workingDirectory: string, form: TrialJobApplicationForm) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.url = `file://localhost:${workingDirectory}`;
    }
}

/**
 * Find the state of a slurm job.
 */
async function getState(slurmJobId: any): Promise<string> {
    const deferred: Deferred<string> = new Deferred<string>();
    let state: string = 'ERROR';
    if (process.platform === 'win32') {
        throw new Error('Windows is not yet supported for nni slurm.');
    }
    else {
        try {
            const result: cpp.childProcessPromise.Result = await cpp.exec(`sacct -j ${slurmJobId} --noheader -o state%30 | head -n 1`);
            state = result.stdout.trim().split(' ')[0];
            if (state === '') state = 'EMPTY'; // Empty query
        } catch (error) {
            //ignore
        }
    }
    deferred.resolve(state);
    return deferred.promise;
}

/**
 * Find the ID of a slurm job.
 */
async function getJobID(jobName: string, testonly: boolean): Promise<number> {
    const deferred: Deferred<number> = new Deferred<number>();
    let slurmJobId: number = -1;
    if (process.platform === 'win32') {
        throw new Error('Windows is not yet supported for nni slurm.');
    }
    else {
        try {
            const startTime = Date.now();
            let lines: string[] = [];
            while (slurmJobId === -1) {
                try {
                    const result: cpp.childProcessPromise.Result = await cpp.exec(`sacct -o jobname%30,jobid%30 -S 0000-01-01 | grep ${jobName}`);
                    lines = result.stdout.split('\n');
                } catch (error) {
                    // ignore empty query
                }
                // get the most recent valid line
                for (let index = 0; index < lines.length; index++) {
                    const line = lines[index];
                    if (line.length > 30) {
                        const match: RegExpMatchArray | null = line.slice(30, 61).match(/\d+/g);
                        if (match) slurmJobId = parseInt(match[0]);
                    }
                }
                if (testonly) {
                    break;
                }
                if (Date.now() - startTime > 60000) {
                    throw new Error('srun does not receive a jobid after 1 min.');
                }
                await delay(500);
            }
        } catch (error) {
            //ignore
        }
    }
    deferred.resolve(slurmJobId);
    return deferred.promise;
}

/**
 * Training Service implementation for slurm (Linux)
 */
class SlurmTrainingService implements TrainingService {
    private readonly config: SlurmConfig;
    private readonly eventEmitter: EventEmitter;
    private readonly jobMap: Map<string, SlurmTrialJobDetail>;
    private readonly jobQueue: string[];
    private initialized: boolean;
    private stopping: boolean;
    private rootDir!: string;
    private readonly experimentId!: string;
    private readonly log: Logger;
    private readonly jobStreamMap: Map<string, ts.Stream>;

    constructor(config: SlurmConfig) {
        this.config = config;
        this.eventEmitter = new EventEmitter();
        this.jobMap = new Map<string, SlurmTrialJobDetail>();
        this.jobQueue = [];
        this.stopping = false;
        this.log = getLogger('SlurmTrainingService');
        this.experimentId = getExperimentId();
        this.jobStreamMap = new Map<string, ts.Stream>();
        this.log.info('Construct slurm machine training service.');

        // slurm resource might be more than gpus
        // this.slurmScheduler = new SlurmScheduler();

        this.rootDir = getExperimentRootDir();
        if (!fs.existsSync(this.rootDir)) {
            throw new Error('root dir not created');
        }
        this.initialized = true;
    }

    public async run(): Promise<void> {
        this.log.info('Run slurm machine training service.');
        if (this.config.useWandb) {
            await execMkdir(path.join(this.rootDir, 'wandb'), true);
        }
        const longRunningTasks: Promise<void>[] = [this.runJobLoop()];
        await Promise.all(longRunningTasks);
        this.log.info('Slurm machine training service exit.');
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];
        for (const key of this.jobMap.keys()) {
            const trialJob: TrialJobDetail = await this.getTrialJob(key);
            jobs.push(trialJob);
        }

        return jobs;
    }

    public async getTrialJob(trialJobId: string): Promise<SlurmTrialJobDetail> {
        const trialJob: SlurmTrialJobDetail | undefined = this.jobMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, 'Trial job not found');
        }
        if (trialJob.slurmJobId === undefined) {
            this.log.debug(`trialJob ${trialJobId} has no slurm job id.`);
        } else if (trialJob.status === 'RUNNING') {
            const state: string = await getState(trialJob.slurmJobId);

            // touch metric file to trigger tail-stream
            // will be invoked every 5 seconds by nnimanager
            if (state !== 'EMPTY') {
                await cpp.exec(`touch ${path.join(trialJob.workingDirectory, '.nni', 'metrics')}`);
            }

            if (state !== 'RUNNING') {
                await delay(1000); // avoid last trial emit failure
                trialJob.endTime = Date.now();
                this.setTrialJobStatus(trialJob, 'FAILED');
                try {
                    const state: string = await fs.promises.readFile(path.join(trialJob.workingDirectory, '.nni', 'state'), 'utf8');
                    const match: RegExpMatchArray | null = state.trim()
                        .match(/^(\d+)\s+(\d+)/);
                    if (match !== null) {
                        const { 1: code, 2: timestamp } = match;
                        if (parseInt(code, 10) === 0) {
                            this.setTrialJobStatus(trialJob, 'SUCCEEDED');
                        }
                        trialJob.endTime = parseInt(timestamp, 10);
                    }
                } catch (error) {
                    //ignore
                }
                this.log.debug(`trialJob status update: ${trialJobId}, ${trialJob.status}`);
            }
        } else if (trialJob.status === 'WAITING') {
            // update slurm job to running
            const slurmState: string = await getState(trialJob.slurmJobId);
            if (slurmState !== 'PENDING' && slurmState !== 'EMPTY') {
                this.setTrialJobStatus(trialJob, 'RUNNING');
                trialJob.startTime = Date.now();
            }
        }

        return trialJob;
    }

    public async getTrialFile(trialJobId: string, fileName: string): Promise<string | Buffer> {
        // check filename here for security
        if (!['trial.log', 'stderr', 'model.onnx', 'stdout', 'slurm_stderr', 'slurm_stdout'].includes(fileName)) {
            throw new Error(`File unaccessible: ${fileName}`);
        }
        let encoding: string | null = null;
        if (!fileName.includes('.') || fileName.match(/.*\.(txt|log)/g)) {
            encoding = 'utf8';
        }
        const logPath = path.join(this.rootDir, 'trials', trialJobId, fileName);
        if (!fs.existsSync(logPath)) {
            throw new Error(`File not found: ${logPath}`);
        }
        return fs.promises.readFile(logPath, {encoding: encoding as any});
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.eventEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.eventEmitter.off('metric', listener);
    }

    public submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobId: string = form.id === undefined ? uniqueString(5) : form.id;
        const trialJobDetail: SlurmTrialJobDetail = new SlurmTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            path.join(this.rootDir, 'trials', trialJobId),
            form
        );
        this.jobQueue.push(trialJobId);
        this.jobMap.set(trialJobId, trialJobDetail);

        this.log.debug('submitTrialJob: return:',  trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    /**
     * Update trial job for multi-phase
     * @param trialJobId trial job id
     * @param form job application form
     */
    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: undefined | TrialJobDetail = this.jobMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        await this.writeParameterFile(trialJobDetail.workingDirectory, form.hyperParameters);

        return trialJobDetail;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJob: SlurmTrialJobDetail | undefined = this.jobMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, 'Trial job not found');
        }
        if (trialJob.slurmJobId === undefined) { // TODO: query with sacct
            this.setTrialJobStatus(trialJob, 'USER_CANCELED');
            return Promise.resolve();
        }
        await cpp.exec(`scancel ${trialJob.slurmJobId}`);
        this.setTrialJobStatus(trialJob, getJobCancelStatus(isEarlyStopped));

        const startTime = Date.now();
        while(await getState(trialJob.slurmJobId) !== 'CANCELLED') {
            if (Date.now() - startTime > 14999) {
                if (await getState(trialJob.slurmJobId) === 'RUNNING') {
                    await cpp.exec(`scancel ${trialJob.slurmJobId}`);
                }
                break;
            }
            await delay(500);
        }

        return Promise.resolve();
    }

    public async setClusterMetadata(_key: string, _value: string): Promise<void> { return; }
    public async getClusterMetadata(_key: string): Promise<string> { return ''; }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping slurm machine training service...');
        this.stopping = true;
        for (const stream of this.jobStreamMap.values()) {
            stream.end(0);
            stream.emit('end');
        }

        // important: stop all trials
        const jobs: SlurmTrialJobDetail[] = await this.listTrialJobs();
        for (const job of jobs) {
            if (job.status === 'RUNNING') {
                await cpp.exec(`scancel ${job.slurmJobId}`);
            }
        }
        if (this.config.useWandb && fs.existsSync(path.join(this.rootDir, 'wandb', 'latest-run'))) {
            this.log.debug(`Upload ${path.join(this.rootDir, 'wandb/offline')}-* to wandb`);
            await cpp.exec(`wandb sync --include-offline ${path.join(this.rootDir, 'wandb/offline')}-*`);
        }

        return Promise.resolve();
    }

    private onTrialJobStatusChanged(trialJob: SlurmTrialJobDetail, oldStatus: TrialJobStatus): void {
        //if job is not running, destory job stream
        if (['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED', 'EARLY_STOPPED'].includes(trialJob.status)) {
            if (this.jobStreamMap.has(trialJob.id)) {
                const stream: ts.Stream | undefined = this.jobStreamMap.get(trialJob.id);
                if (stream === undefined) {
                    throw new Error(`Could not find stream in trial ${trialJob.id}`);
                }
                //Refer https://github.com/Juul/tail-stream/issues/20
                setTimeout(() => {
                    stream.end(0);
                    stream.emit('end');
                    this.jobStreamMap.delete(trialJob.id);

                    if (this.config.useWandb && fs.existsSync(path.join(this.rootDir, 'wandb', 'latest-run'))) {
                        this.log.debug(`Upload ${path.join(this.rootDir, 'wandb')} to wandb`);
                        cpp.exec(`(cd ${this.rootDir}; wandb sync --no-include-synced --include-offline --sync-all)`);
                    }
                }, 5000);
            }
        }
        this.log.debug(`trialJob status update: ${trialJob.id}, ${oldStatus} -> ${trialJob.status}`);
    }

    private getEnvironmentVariables( trialJobDetail: TrialJobDetail ): { key: string; value: string }[] {
        let envVariables: { key: string; value: string }[] = [
            { key: 'NNI_PLATFORM', value: 'slurm' },
            { key: 'NNI_EXP_ID', value: this.experimentId },
            { key: 'NNI_SYS_DIR', value: trialJobDetail.workingDirectory },
            { key: 'NNI_TRIAL_JOB_ID', value: trialJobDetail.id },
            { key: 'NNI_OUTPUT_DIR', value: trialJobDetail.workingDirectory },
            { key: 'NNI_TRIAL_SEQ_ID', value: trialJobDetail.form.sequenceId.toString() },
            { key: 'NNI_CODE_DIR', value: this.config.trialCodeDirectory}
        ];

        if (this.config.useWandb) {
            envVariables = envVariables.concat([
                { key: 'USE_WANDB_NNI', value: 'true' },
                { key: 'WANDB_MODE', value: 'dryrun' }
            ]);
        }

        return envVariables;
    }

    private async runJobLoop(): Promise<void> {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length !== 0) {
                const trialJobId: string = this.jobQueue[0];
                const trialJobDetail: SlurmTrialJobDetail | undefined = this.jobMap.get(trialJobId);
                if (trialJobDetail !== undefined && trialJobDetail.status === 'WAITING' && trialJobDetail.slurmJobId === undefined) {
                    await this.runTrialJob(trialJobId);
                }
                this.jobQueue.shift();
            }
            await delay(5000);
        }
    }

    private setTrialJobStatus(trialJob: SlurmTrialJobDetail, newStatus: TrialJobStatus): void {
        if (trialJob.status !== newStatus) {
            const oldStatus: TrialJobStatus = trialJob.status;
            trialJob.status = newStatus;
            this.onTrialJobStatusChanged(trialJob, oldStatus);
        }
    }

    private getScript(workingDirectory: string, trialCommand: string): string[] {
        const script: string[] = [];
        const escapedCommand = shellString(trialCommand);
        if (process.platform === 'win32') {
            script.push(`$PSDefaultParameterValues = @{'Out-File:Encoding' = 'utf8'}`);
            script.push(`cd $env:NNI_CODE_DIR`);
            script.push(
                `cmd.exe /c ${escapedCommand} 1>${path.join(workingDirectory, 'stdout')} 2>${path.join(workingDirectory, 'stderr')}`,
                `$NOW_DATE = [int64](([datetime]::UtcNow)-(get-date "1/1/1970")).TotalSeconds`,
                `$NOW_DATE = "$NOW_DATE" + (Get-Date -Format fff).ToString()`,
                `Write $LASTEXITCODE " " $NOW_DATE  | Out-File "${path.join(workingDirectory, '.nni', 'state')}" -NoNewline -encoding utf8`);
        } else {
            script.push(`cd $NNI_CODE_DIR`);
            script.push(`eval ${escapedCommand} 1>${path.join(workingDirectory, 'stdout')} 2>${path.join(workingDirectory, 'stderr')}`);
            if (process.platform === 'darwin') {
                // https://superuser.com/questions/599072/how-to-get-bash-execution-time-in-milliseconds-under-mac-os-x
                // Considering the worst case, write 999 to avoid negative duration
                script.push(`echo $? \`date +%s999\` >'${path.join(workingDirectory, '.nni', 'state')}'`);
            } else {
                script.push(`echo $? \`date +%s%3N\` >'${path.join(workingDirectory, '.nni', 'state')}'`);
            }
        }

        return script;
    }

    private getSrunCommand(trialJobId: string, workingDirectory: string, config: { [key: string]: string | null }): string[] {
        const command: string[] = [];
        command.push(`srun`);
        command.push(`'--job-name=nni-${this.experimentId}-${trialJobId}'`);
        command.push(`'--output=${path.join(workingDirectory, 'slurm_stdout')}'`);
        command.push(`'--error=${path.join(workingDirectory, 'slurm_stderr')}'`);
        command.push(`'--disable-status'`);
        command.push(`'--unbuffered'`);
        Object.entries(config).forEach(
            ([key, value]) => {
                const prefix: string = (key.length === 1 ? '-' : '--');
                if (value === null) {
                    command.push(`'${prefix}${key}'`); // no argument
                } else {
                    command.push(`'${prefix}${key}=${value}'`);
                }
            }
        );
        command.push(this.config.trialCommand);

        return command;
    }

    private getSbatchHeader(trialJobId: string, workingDirectory: string, config: { [key: string]: string | null }): string[] {
        const script: string[] = [];
        script.push(`#SBATCH --job-name=nni-${this.experimentId}-${trialJobId}`);
        script.push(`#SBATCH --output=${path.join(workingDirectory, 'slurm_stdout')}`);
        script.push(`#SBATCH --error=${path.join(workingDirectory, 'slurm_stderr')}`);
        Object.entries(config).forEach(
            ([key, value]) => {
                const prefix: string = (key.length === 1 ? '-' : '--');
                if (value === null) {
                    script.push(`#SBATCH ${prefix}${key}`); // no argument
                } else {
                    script.push(`#SBATCH ${prefix}${key}=${value}`);
                }
            }
        );
        script.push(``); // add an empty line after header

        return script;
    }

    private async runTrialJob(trialJobId: string): Promise<void> {
        const trialJobDetail: SlurmTrialJobDetail = <SlurmTrialJobDetail>this.jobMap.get(trialJobId);
        const variables: { key: string; value: string }[] = this.getEnvironmentVariables(trialJobDetail);

        const runScriptContent: string[] = [];
        if (process.platform !== 'win32') {
            runScriptContent.push('#!/bin/bash');
        } else {
            runScriptContent.push(`$env:PATH=${powershellString(process.env['path']!)}`)
        }

        // set sbatch header
        if (this.config.useSbatch) {
            const headers: string[] = this.getSbatchHeader(trialJobId, trialJobDetail.workingDirectory, this.config.resource);
            headers.forEach((script: string) => {
                runScriptContent.push(script);
            });
        }

        // set environment variables
        for (const variable of variables) {
            runScriptContent.push(setEnvironmentVariable(variable));
        }

        // generate script files content
        const trialCommand: string = this.config.useSbatch? this.config.trialCommand : this.getSrunCommand(trialJobId, trialJobDetail.workingDirectory, this.config.resource).join(' ');
        const scripts: string[] = this.getScript(trialJobDetail.workingDirectory, trialCommand);
        scripts.forEach((script: string) => {
            runScriptContent.push(script);
        });

        const slurmJobId: number = await getJobID(`nni-${this.experimentId}-${trialJobId}`, true);
        if (slurmJobId === -1) {
            // prepare to execute
            await execMkdir(trialJobDetail.workingDirectory);
            await execMkdir(path.join(trialJobDetail.workingDirectory, '.nni'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, '.nni', 'metrics'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, 'stdout'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, 'stderr'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, 'slurm_stdout'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, 'slurm_stderr'));
            const scriptName: string = getScriptName('run');
            await createScriptFile(path.join(trialJobDetail.workingDirectory, scriptName),
                    runScriptContent.join(getNewLine()));
            await this.writeParameterFile(trialJobDetail.workingDirectory, trialJobDetail.form.hyperParameters);
    
            // execute the trial
            if (this.config.useSbatch) {
                cp.exec(`sbatch '${path.join(trialJobDetail.workingDirectory, scriptName)}'`);
            } else {
                cp.exec(`bash '${path.join(trialJobDetail.workingDirectory, scriptName)}'`);
            }

            // obtain the slurm job ID
            trialJobDetail.slurmJobId = await getJobID(`nni-${this.experimentId}-${trialJobId}`, false);
        } else {
            // the job has been executed before
            await execMkdir(trialJobDetail.workingDirectory);
            await execMkdir(path.join(trialJobDetail.workingDirectory, '.nni'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, '.nni', 'metrics'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, 'stdout'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, 'stderr'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, 'slurm_stdout'));
            await execNewFile(path.join(trialJobDetail.workingDirectory, 'slurm_stderr'));
            trialJobDetail.slurmJobId = slurmJobId;
        }

        this.log.debug(`trial ${trialJobId} gets jobid ${trialJobDetail.slurmJobId}`);
        this.setTrialJobStatus(trialJobDetail, 'WAITING');
        trialJobDetail.startTime = Date.now(); // eslint-disable-line require-atomic-updates

        let buffer: Buffer = Buffer.alloc(0);
        const stream: ts.Stream = ts.createReadStream(path.join(trialJobDetail.workingDirectory, '.nni', 'metrics'));
        stream.on('data', (data: Buffer) => {
            buffer = Buffer.concat([buffer, data]);
            while (buffer.length > 0) {
                const [success, , content, remain] = decodeCommand(buffer);
                if (!success) {
                    break;
                }
                this.eventEmitter.emit('metric', {
                    id: trialJobDetail.id,
                    data: content
                });
                this.log.debug(`Sending metrics, job id: ${trialJobDetail.id}, metrics: ${content}`);
                buffer = remain;
            }
        });
        this.jobStreamMap.set(trialJobDetail.id, stream);
    }

    public get MetricsEmitter(): EventEmitter {
        return this.eventEmitter;
    }

    private async writeParameterFile(directory: string, hyperParameters: HyperParameters): Promise<void> {
        const filepath: string = path.join(directory, generateParamFileName(hyperParameters));
        await fs.promises.writeFile(filepath, hyperParameters.value, { encoding: 'utf8' });
    }

    public async getTrialOutputLocalPath(trialJobId: string): Promise<string> {
        return Promise.resolve(path.join(this.rootDir, 'trials', trialJobId));
    }

    public async fetchTrialOutput(trialJobId: string, subpath: string): Promise<void> {
        let trialLocalPath = await this.getTrialOutputLocalPath(trialJobId);
        if (subpath !== undefined) {
            trialLocalPath = path.join(trialLocalPath, subpath);
        }
        if (fs.existsSync(trialLocalPath)) {
            return Promise.resolve();
        } else {
            return Promise.reject(new Error('Trial local path not exist.'));
        }
    }
}

export { SlurmTrainingService };
