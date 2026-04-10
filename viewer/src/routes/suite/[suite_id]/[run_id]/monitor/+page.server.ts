import type { PageServerLoad } from './$types.js';

export const load: PageServerLoad = async ({ params }) => {
	return {
		suite_id: params.suite_id,
		run_id: params.run_id
	};
};
