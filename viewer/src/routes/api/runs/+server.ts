import { json } from '@sveltejs/kit';
import { getActiveRuns } from '$lib/server/runner.js';
import type { RequestHandler } from './$types.js';

export const GET: RequestHandler = async () => {
	const runs = getActiveRuns().map((r) => ({
		suiteId: r.suiteId,
		runId: r.runId,
		status: r.status,
		startedAt: r.startedAt,
		currentStage: r.currentStage,
		stages: r.stages
	}));
	return json(runs);
};
