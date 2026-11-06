import json
import re
import numpy as np
import torch
from PIL import Image

VLM4TS_SYSTEM_PROMPT = """You are an expert industrial reliability engineer inspecting multivariate sensor telemetry and actuator plots from a critical infrastructure facility (e.g. water treatment plant).
The image has two columns:
- LEFT COLUMN: Historical macro baseline (prior normal operational cycle).
- RIGHT COLUMN: Target micro window under inspection (relative time steps 0 to 200).
- TOP ROW: Continuous analog process metrics (z-standardized levels, flow rates, pressure, conductivity).
- BOTTOM ROW: Actuator states (valves, pumps: 1=closed/off, 2=open/on) or process rate of change (Δz).

DIAGNOSTIC PRINCIPLES (LOOK BEYOND SIMPLE HIGH-AMPLITUDE SPIKES):
1. NORMAL OPERATION:
   - Analog signals fluctuate smoothly within the baseline operating range.
   - Actuators toggle periodically with corresponding expected changes in downstream levels and flows.
2. ANOMALIES & ATTACKS:
   - INVARIANT VIOLATION: An actuator turns ON or valve opens, but downstream flow/level does not respond, or remains zero.
   - STUCK-AT / SENSOR FREEZE: A sensor signal becomes unnaturally flat or holds a constant horizontal level inside the normal band when it should be fluctuating with process cycles.
   - LEVEL OFFSET / BIAS: A sustained shift or plateau in tank levels or pressures that deviates from baseline cycle periodicity, even if within ±3σ.
   - DRIFT / RAMPS: Slow continuous divergence or ramping away from expected nominal behavior.

Answer strictly with a JSON object:
{
  "is_anomaly": true or false,
  "confidence": 1 to 5,
  "anomaly_likelihood": 0.0 to 1.0,
  "intervals": [[start_step, end_step]],
  "anomaly_type": "invariant_violation" | "stuck_at" | "step_shift" | "drift" | "spike" | "none",
  "reason": "Clear explanation of the physical inconsistency observed"
}"""


class VLM4TSReasoner:
    """Stage 2 of VLM4TS: Multimodal verification and structured semantic reasoning.
    
    Supports Qwen2.5-VL and Chart-tailored VLMs for deep temporal-visual inspection,
    producing calibrated likelihoods, localized intervals, and dense teacher guidance masks.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """Lazy loads the multimodal VLM backbone."""
        if self.model is not None:
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print(f"Loading VLM4TS Reasoner backbone: {self.model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def inspect_window(self, pil_image: Image.Image, window_size=200) -> dict:
        """Runs multi-modal reasoning on the rendered dual-horizon image."""
        self.load_model()
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": VLM4TS_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Analyze the target micro window (steps 0 to 200). Is there an operational anomaly, attack, or invariant violation? Output JSON only."}
            ]}
        ]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        gen_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response_text = self.processor.decode(gen_tokens, skip_special_tokens=True).strip()

        # Extract latent multimodal visual feature from the last prompt hidden state
        if outputs.hidden_states:
            last_hidden = outputs.hidden_states[0][-1]  # [1, seq_len, hidden_dim]
            z_feature = last_hidden[:, -1, :].squeeze(0).float().cpu()
        else:
            z_feature = torch.zeros(2048, dtype=torch.float32)

        # Parse JSON response
        parsed = self._parse_json_response(response_text)
        a_T = float(parsed.get("anomaly_likelihood", 0.8 if parsed.get("is_anomaly") else 0.2))

        # Build dense mask over the window
        dense_mask = np.zeros(window_size, dtype=np.float32)
        intervals = parsed.get("intervals", [])
        if intervals:
            for iv in intervals:
                if isinstance(iv, (list, tuple)) and len(iv) == 2:
                    st, ed = max(0, int(iv[0])), min(window_size, int(iv[1]))
                    if ed > st:
                        dense_mask[st:ed] = a_T
        elif parsed.get("is_anomaly"):
            dense_mask[:] = a_T

        return {
            "is_anomaly": parsed.get("is_anomaly", a_T >= 0.5),
            "anomaly_confidence": int(parsed.get("confidence", 3)),
            "anomaly_likelihood": a_T,
            "anomaly_type": parsed.get("anomaly_type", "unknown"),
            "intervals": intervals,
            "reason": parsed.get("reason", response_text[:120]),
            "z_feature": z_feature,
            "teacher_dense_mask": torch.from_numpy(dense_mask)
        }

    def _parse_json_response(self, text: str) -> dict:
        """Robust JSON extraction from LLM text output."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        is_anom = bool("true" in text.lower() or "anomaly" in text.lower())
        return {
            "is_anomaly": is_anom,
            "confidence": 3 if is_anom else 1,
            "anomaly_likelihood": 0.85 if is_anom else 0.15,
            "intervals": [[0, 200]] if is_anom else [],
            "anomaly_type": "detected" if is_anom else "none",
            "reason": text[:150]
        }
