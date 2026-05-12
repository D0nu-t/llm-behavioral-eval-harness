from eval_harness.probes.sycophancy import OpinionAssertionProbe
from eval_harness.probes.jailbreak import JailbreakProbe
from eval_harness.probes.hallucination import HallucinationProbe
from eval_harness.probes.refusal import RefusalProbe
from eval_harness.probes.bias import BiasProbe
from eval_harness.probes.instruction_following import InstructionFollowingProbe
from eval_harness.probes.safety import SafetyProbe


PROBE_REGISTRY = {
    "sycophancy": OpinionAssertionProbe,
    "jailbreak": JailbreakProbe,
    "hallucination": HallucinationProbe,
    "refusal": RefusalProbe,
    "bias": BiasProbe,
    "instruction_following": InstructionFollowingProbe,
    "safety": SafetyProbe,
}
