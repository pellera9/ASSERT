import { loadComparePageData } from '$lib/server/data.js';
import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types.js';

export const load: PageServerLoad = async ({ params, url }) => {
	const { suite_id } = params;
	const runsParam = url.searchParams.get('runs');

	if (!runsParam) throw error(400, 'Missing "runs" query parameter');

	const runIds = runsParam.split(',').filter(Boolean);
	if (runIds.length < 2) throw error(400, 'Need at least 2 runs to compare');
	if (runIds.length > 4) throw error(400, 'Maximum 4 runs to compare');

	const payload = loadComparePageData(suite_id, runIds);
	if (!payload) throw error(404, 'One or more runs had no judged samples');
	return payload;
};
