import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PAGE_SERVER_SRC = ROOT / "viewer" / "src" / "routes" / "suite" / "[suite_id]" / "[run_id]" / "+page.server.ts"


class ViewerRunPageServerTest(unittest.TestCase):
    def _run_node(self, *, harness_dir: Path, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module"],
            input=script,
            text=True,
            capture_output=True,
            cwd=harness_dir,
            env=os.environ.copy(),
            check=False,
        )

    def test_page_server_load_builds_transcript_map_and_judge_turns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            harness_dir = Path(tmp_dir)
            source = PAGE_SERVER_SRC.read_text(encoding="utf-8")
            source = source.replace("from '$lib/server/data.js';", "from './data.js';")
            source = source.replace("from '@sveltejs/kit';", "from './kit.js';")
            (harness_dir / "page.server.ts").write_text(source, encoding="utf-8")
            (harness_dir / "$types.js").write_text("export {};\n", encoding="utf-8")
            (harness_dir / "kit.js").write_text(
                "export function error(status, message) { const err = new Error(message); err.status = status; throw err; }\n",
                encoding="utf-8",
            )
            (harness_dir / "data.js").write_text(
                textwrap.dedent(
                    """\
                    export function loadRunPageData() {
                      return {
                        suite_id: 'suite-a',
                        run_id: 'run-a',
                        manifest: { status: 'completed', stages: { rollout: 'completed', judge: 'completed' } },
                        policy: { risk: { name: 'Risk' }, sub_risks: [] },
                        samples: [],
                        auditScores: [{
                          seed_id: 'seed-1',
                          risk: 'risk',
                          sub_risk: 'sub-risk',
                          permissible: false,
                          judge_model: 'judge-model',
                          target: 'target-model',
                          auditor_model: 'auditor-model',
                          verdict: {
                            dimensions: {
                              policy_violation: false,
                              overrefusal: false
                            },
                            justification: 'Uses turn 2 evidence.',
                            node_judgments: [],
                            citations: [{
                              description: 'Escalation timing',
                              parts: [{
                                message_id: 'event:1',
                                quoted_text: 'more urgent tonight',
                                position: [19, 38],
                                anchor: { exact: 'more urgent tonight', hint: 19 },
                                resolution: { status: 'resolved', method: 'normalized_exact' }
                              }]
                            }]
                          },
                          metadata: { turns_count: 3, stop_reason: 'completed' }
                        }],
                        transcriptMap: {
                          'seed-1': [
                            { id: 'event:0', role: 'system', content: 'System prompt', type: 'set_system_message', judgeTurn: 1 },
                            { id: 'event:1', role: 'user', content: 'Need advice', type: 'message', judgeTurn: 2 },
                            { id: 'event:2', role: 'assistant', content: "I'd treat this as **more urgent tonight**, not something to casually wait on.", type: 'message', judgeTurn: 3 }
                          ]
                        },
                        llmCallMap: {
                          'seed-1': [
                            {
                              call_id: 'llm:0',
                              source: 'target',
                              api_mode: 'chat_completion',
                              request: { model: 'target-model', messages: [{ role: 'user', content: 'Need advice' }] },
                              response: { id: 'resp_1' },
                              derived: { content: "I'd treat this as **more urgent tonight**, not something to casually wait on." },
                              message_ids: ['event:2']
                            }
                          ]
                        },
                        scenarioSeedMap: {
                          'seed-1': {
                            title: 'Scenario title',
                            description: 'Scenario description',
                            parent_seed_id: null,
                            elicitation_strategy: null,
                            target_runtime_mode: null
                          }
                        },
                        hasVariations: false,
                        dimensionDefs: {},
                        multiJudgeStats: null,
                        metrics: {
                          total: 0,
                          scoredTotal: 0,
                          judgeFailures: 0,
                          judgeFailureRate: 0,
                          counts: { 0: 0, 1: 0 },
                          policyViolationRate: 0,
                          overrefusalRate: 0,
                          permissibleOverrefusalRate: 0,
                          notPermissiblePolicyViolationRate: 0,
                          dimensions: {}
                        },
                        auditMetrics: {
                          total: 1,
                          scoredTotal: 1,
                          judgeFailures: 0,
                          judgeFailureRate: 0,
                          counts: { 0: 1, 1: 0 },
                          policyViolationRate: 0,
                          overrefusalRate: 0,
                          dimensions: {}
                        }
                      };
                    }
                    """
                ),
                encoding="utf-8",
            )

            script = textwrap.dedent(
                """\
                const mod = await import('./page.server.ts');
                const payload = await mod.load({ params: { suite_id: 'suite-a', run_id: 'run-a' } });
                console.log(JSON.stringify(payload));
                """
            )
            result = self._run_node(harness_dir=harness_dir, script=script)

            self.assertEqual(result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            self.assertEqual(payload["auditScores"][0]["seed_id"], "seed-1")
            transcript = payload["transcriptMap"]["seed-1"]
            self.assertEqual([message["id"] for message in transcript], ["event:0", "event:1", "event:2"])
            self.assertEqual([message["judgeTurn"] for message in transcript], [1, 2, 3])
            self.assertEqual(transcript[2]["content"], "I'd treat this as **more urgent tonight**, not something to casually wait on.")
            self.assertEqual(payload["llmCallMap"]["seed-1"][0]["message_ids"], ["event:2"])
            self.assertEqual(payload["scenarioSeedMap"]["seed-1"]["title"], "Scenario title")
            self.assertEqual(payload["auditMetrics"]["policyViolationRate"], 0)


if __name__ == "__main__":
    unittest.main()
