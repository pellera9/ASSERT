import { loadRunPageData } from '$lib/server/data.js';
import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types.js';

export const load: PageServerLoad = async ({ params }) => {
	const payload = loadRunPageData(params.suite_id, params.run_id);
	if (!payload) throw error(404, `Run "${params.run_id}" not found in suite "${params.suite_id}"`);
	return payload;
};
