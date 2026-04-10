import fs from 'node:fs';
import path from 'node:path';
import { parse as parseYaml } from 'yaml';
import { ARTIFACTS_ROOT } from './config.js';
import type { Manifest, Policy, Suite } from '$lib/types.js';

export const SUITE_SEEDS_FILE = 'seeds.jsonl';
export const RUN_TRANSCRIPTS_FILE = 'transcripts.jsonl';
export const RUN_SCORES_FILE = 'scores.jsonl';
export const RUN_CONFIG_FILE = 'config.yaml';
export const RUN_MANIFEST_FILE = 'manifest.json';
export const SUITE_METADATA_FILE = 'suite.json';
export const SUITE_POLICY_FILE = 'policy.json';
export const SUITE_SYSTEMATIZATION_FILE = 'systematization.json';

const SAFE_ID_RE = /^[a-z0-9][a-z0-9._-]*$/i;

export type UnifiedSeedRow = Record<string, unknown> & {
	kind?: unknown;
	seed_id?: unknown;
	seed?: unknown;
};

export type UnifiedTranscriptRow = Record<string, unknown> & {
	kind?: unknown;
	seed_id?: unknown;
	events?: unknown;
	llm_calls?: unknown;
	stop_reason?: unknown;
	risk?: unknown;
	sub_risk?: unknown;
	permissible?: unknown;
	target?: unknown;
	auditor_model?: unknown;
};

export type UnifiedScoreRow = Record<string, unknown> & {
	kind?: unknown;
	seed_id?: unknown;
	verdict?: unknown;
	judge_status?: unknown;
	judge_error?: unknown;
	target?: unknown;
	auditor_model?: unknown;
};

export interface SuiteSnapshot {
	suiteId: string;
	suiteDir: string;
	suite: Suite | null;
	policy: Policy | null;
	seedRows: UnifiedSeedRow[];
	runIds: string[];
	systematization: Record<string, unknown> | null;
}

export interface RunSnapshot {
	suiteId: string;
	runId: string;
	runDir: string;
	manifest: Manifest | null;
	config: Record<string, unknown> | null;
	seedRows: UnifiedSeedRow[];
	scoreRows: UnifiedScoreRow[];
	transcriptRows: UnifiedTranscriptRow[];
	runtimeMode: string | null;
}

export class ArtifactParseError extends Error {
	filePath: string;
	format: 'json' | 'jsonl' | 'yaml';

	constructor(
		filePath: string,
		format: 'json' | 'jsonl' | 'yaml',
		message: string,
		options?: { cause?: unknown }
	) {
		super(message, options);
		this.name = 'ArtifactParseError';
		this.filePath = filePath;
		this.format = format;
	}
}

function isMissingError(error: unknown): boolean {
	return Boolean(error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT');
}

function readTextFile(filePath: string, { missingOk = false }: { missingOk?: boolean } = {}): string | null {
	try {
		return fs.readFileSync(filePath, 'utf-8');
	} catch (error) {
		if (missingOk && isMissingError(error)) return null;
		throw error;
	}
}

export function readJsonFile<T>(
	filePath: string,
	{ missingOk = false }: { missingOk?: boolean } = {}
): T | null {
	const text = readTextFile(filePath, { missingOk });
	if (text === null) return null;
	try {
		return JSON.parse(text) as T;
	} catch (error) {
		throw new ArtifactParseError(filePath, 'json', `Invalid JSON in ${filePath}`, { cause: error });
	}
}

export function readJsonlFile<T>(
	filePath: string,
	{ missingOk = false }: { missingOk?: boolean } = {}
): T[] {
	const text = readTextFile(filePath, { missingOk });
	if (text === null) return [];
	const trimmed = text.trim();
	if (!trimmed) return [];
	try {
		return trimmed.split('\n').map((line, index) => {
			try {
				return JSON.parse(line) as T;
			} catch (error) {
				throw new ArtifactParseError(
					filePath,
					'jsonl',
					`Invalid JSONL in ${filePath} on line ${index + 1}`,
					{ cause: error }
				);
			}
		});
	} catch (error) {
		if (error instanceof ArtifactParseError) throw error;
		throw new ArtifactParseError(filePath, 'jsonl', `Invalid JSONL in ${filePath}`, { cause: error });
	}
}

export function readLiveTranscriptJsonlFile<T>(
	filePath: string,
	{ missingOk = false }: { missingOk?: boolean } = {}
): T[] {
	const text = readTextFile(filePath, { missingOk });
	if (text === null) return [];
	if (!text.trim()) return [];

	const hasTrailingNewline = text.endsWith('\n');
	const segments = text.split('\n');
	if (hasTrailingNewline && segments[segments.length - 1] === '') {
		segments.pop();
	}

	const rows: T[] = [];
	for (const [index, line] of segments.entries()) {
		const isFinalSegment = index === segments.length - 1;
		try {
			rows.push(JSON.parse(line) as T);
		} catch (error) {
			if (isFinalSegment && !hasTrailingNewline) {
				break;
			}
			throw new ArtifactParseError(
				filePath,
				'jsonl',
				`Invalid JSONL in ${filePath} on line ${index + 1}`,
				{ cause: error }
			);
		}
	}

	return rows;
}

export function readYamlFile<T>(
	filePath: string,
	{ missingOk = false }: { missingOk?: boolean } = {}
): T | null {
	const text = readTextFile(filePath, { missingOk });
	if (text === null) return null;
	try {
		return parseYaml(text) as T;
	} catch (error) {
		throw new ArtifactParseError(filePath, 'yaml', `Invalid YAML in ${filePath}`, { cause: error });
	}
}

export function listSubdirectories(dirPath: string): string[] {
	try {
		return fs
			.readdirSync(dirPath, { withFileTypes: true })
			.filter((entry) => entry.isDirectory())
			.map((entry) => entry.name);
	} catch (error) {
		if (isMissingError(error)) return [];
		throw error;
	}
}

export function isSafeArtifactId(id: string): boolean {
	return SAFE_ID_RE.test(id) && !id.includes('..');
}

export function suiteDirPath(suiteId: string): string {
	return path.join(ARTIFACTS_ROOT, suiteId);
}

export function runDirPath(suiteId: string, runId: string): string {
	return path.join(suiteDirPath(suiteId), runId);
}

export function resolveArtifactPath(requestPath: string): string {
	const artifactsRoot = path.resolve(ARTIFACTS_ROOT);
	const resolvedPath = path.resolve(artifactsRoot, requestPath);
	const relativePath = path.relative(artifactsRoot, resolvedPath);
	if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
		throw new Error('Artifact path escaped artifacts root');
	}
	return resolvedPath;
}

function readObject(value: unknown): Record<string, unknown> | null {
	return value && typeof value === 'object' && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: null;
}

export function loadRunRuntimeMode(config: Record<string, unknown> | null): string | null {
	const pipeline = readObject(config?.pipeline);
	const rollout = readObject(pipeline?.rollout);
	const target = readObject(rollout?.target);
	const tools = readObject(target?.tools);

	if (typeof target?.connector === 'string' && target.connector) return 'external';
	if (typeof tools?.module === 'string' && tools.module) return 'tool_module';
	if (typeof tools?.toolset === 'string' && tools.toolset) return 'simulated';

	const targetModel = readObject(target?.model);
	if (typeof targetModel?.name === 'string' && targetModel.name) return 'chat';
	return null;
}

export function loadSuiteSnapshot(suiteId: string): SuiteSnapshot | null {
	const suiteDir = suiteDirPath(suiteId);
	const suite = readJsonFile<Suite>(path.join(suiteDir, SUITE_METADATA_FILE), { missingOk: true });
	const policy = readJsonFile<Policy>(path.join(suiteDir, SUITE_POLICY_FILE), { missingOk: true });
	if (!suite && !policy) return null;

	return {
		suiteId,
		suiteDir,
		suite,
		policy,
		seedRows: readJsonlFile<UnifiedSeedRow>(path.join(suiteDir, SUITE_SEEDS_FILE), { missingOk: true }),
		runIds: listSubdirectories(suiteDir),
		systematization: readJsonFile<Record<string, unknown>>(
			path.join(suiteDir, SUITE_SYSTEMATIZATION_FILE),
			{ missingOk: true }
		)
	};
}

export function loadRunSnapshot(
	suiteId: string,
	runId: string,
	seedRows?: UnifiedSeedRow[]
): RunSnapshot {
	const runDir = runDirPath(suiteId, runId);
	const config = readYamlFile<Record<string, unknown>>(path.join(runDir, RUN_CONFIG_FILE), {
		missingOk: true
	});
	const manifest = readJsonFile<Manifest>(path.join(runDir, RUN_MANIFEST_FILE), { missingOk: true });
	const rolloutRunning = manifest?.stages?.rollout === 'running';

	return {
		suiteId,
		runId,
		runDir,
		manifest,
		config,
		seedRows:
			seedRows ?? readJsonlFile<UnifiedSeedRow>(path.join(suiteDirPath(suiteId), SUITE_SEEDS_FILE), { missingOk: true }),
		scoreRows: readJsonlFile<UnifiedScoreRow>(path.join(runDir, RUN_SCORES_FILE), { missingOk: true }),
		transcriptRows: rolloutRunning
			? readLiveTranscriptJsonlFile<UnifiedTranscriptRow>(path.join(runDir, RUN_TRANSCRIPTS_FILE), {
					missingOk: true
				})
			: readJsonlFile<UnifiedTranscriptRow>(path.join(runDir, RUN_TRANSCRIPTS_FILE), {
					missingOk: true
				}),
		runtimeMode: loadRunRuntimeMode(config)
	};
}
