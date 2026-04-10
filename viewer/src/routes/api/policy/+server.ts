import { json } from '@sveltejs/kit';
import fs from 'node:fs';
import path from 'node:path';
import { ARTIFACTS_ROOT } from '$lib/server/config.js';
import { isSafeArtifactId } from '$lib/server/artifacts.js';
import type { RequestHandler } from './$types.js';

export const PUT: RequestHandler = async ({ request }) => {
	const { suite_id, policy } = await request.json();

	if (typeof suite_id !== 'string' || !suite_id || !policy) {
		return json({ error: 'suite_id and policy are required' }, { status: 400 });
	}

	if (!policy.risk || !Array.isArray(policy.sub_risks)) {
		return json({ error: 'policy must have risk and sub_risks' }, { status: 400 });
	}

	if (!isSafeArtifactId(suite_id)) {
		return json({ error: 'invalid suite_id' }, { status: 400 });
	}

	const suiteDir = path.join(ARTIFACTS_ROOT, suite_id);
	const policyPath = path.join(suiteDir, 'policy.json');

	if (!fs.existsSync(suiteDir)) {
		return json({ error: `Suite "${suite_id}" not found` }, { status: 404 });
	}

	// Write policy
	fs.writeFileSync(policyPath, JSON.stringify(policy, null, 2), 'utf-8');

	return json({ ok: true, sub_risk_count: policy.sub_risks.length });
};
