# Risk Policies

Each `.md` file describes a risk area — what it is and why it matters. The pipeline uses this text to generate a policy taxonomy and evaluate the target model against it.

**To use a risk policy**, set `risk: <filename_without_extension>` in your pipeline config. For example, `risk: harmful_medical_advice` loads `harmful_medical_advice.md`.

**To add your own**, create a new `.md` file here with a clear description of the risk.

## Available policies

| File | Risk area |
|------|-----------|
| `harmful_medical_advice.md` | Harmful medical advice: diagnoses, dosage, treatment plans |
