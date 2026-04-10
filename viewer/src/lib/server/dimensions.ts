import fs from 'node:fs';
import path from 'node:path';
import { stringify as stringifyYaml } from 'yaml';
import { MEASUREMENTS_ROOT } from './config.js';
import { readYamlFile } from './artifacts.js';
import type { DimensionDef } from '$lib/types.js';

const DIMENSIONS_PATH = path.join(
	MEASUREMENTS_ROOT,
	'examples',
	'eval-definitions',
	'judge_dimensions.yaml'
);

export function loadDimensions(): Record<string, DimensionDef> {
	const data = readYamlFile<Record<string, DimensionDef>>(DIMENSIONS_PATH, { missingOk: true });
	return data && typeof data === 'object' ? data : {};
}

export function saveDimension(name: string, description: string, rubric: string): void {
	const dimensions = loadDimensions();
	dimensions[name] = { description, rubric, kind: 'event', polarity: 'negative' };
	fs.writeFileSync(DIMENSIONS_PATH, stringifyYaml(dimensions), 'utf-8');
}
