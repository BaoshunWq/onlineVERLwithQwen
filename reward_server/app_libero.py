# app_libero.py
import sys
sys.path.insert(0, "/root/code/saftyVLA/libero_reward_server")
from flask import Flask, request, jsonify
import traceback
from reward_server.reward_libero import compute_score, DEFAULT_LIBERO_CFG




def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return "ok", 200

    @app.post("/score")
    def score():
        """
        Request JSON（只有 responses，可选 metas）:
        {
          "responses": [ <vLLM对象1>, <vLLM对象2>, ... ],
          "metas": [
            {
              "original_instruction": "put the red bowl on the left shelf",
              "suite": "libero_object",
              "task_id": 3,
              "seed": 0,
              "init_state_id": 0
            },
            ...
          ],
          "reward_function_kwargs": {
            "alpha": 1.0, "beta": 0.1, "gamma": 0.5,
            "num_trials_per_task": 1, "center_crop": 1,
            "libero_cfg": {
              "model_family": "openvla",
              "task_suite_name": "libero_spatial",
              "task_id": 0,
              "checkpoint": "/path/to/openvla.ckpt"
            }
          }
        }
        """
        # try:
        body = request.get_json(force=True) or {}

        # import pdb
        # pdb.set_trace()
        responses = body.get("responses", [])

        metas     = body.get("metas", None)
        kwargs    = body.get("reward_function_kwargs", {}) or {}

        if not isinstance(responses, list):
            return jsonify({"error": "`responses` must be a list"}), 400
        if len(responses) == 0:
            return jsonify({"done_result": []}), 200

        # 填默认 libero_cfg
        kwargs = dict(kwargs)
        lib_cfg = dict(DEFAULT_LIBERO_CFG)
        lib_cfg.update(kwargs.get("libero_cfg", {}) or {})
        kwargs["libero_cfg"] = lib_cfg

        done_result = compute_score(responses, metas, **kwargs)
        return jsonify({"done_result": done_result}), 200

        # except Exception as e:
        #     return jsonify({
        #         "error": str(e),
        #         "traceback": traceback.format_exc(limit=2)
        #     }), 500

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=45678, debug=True)



# # 方式A：Flask内置（调试用）
# python app_libero.py
# # 方式B：gunicorn（推荐）
# gunicorn "app_libero:create_app()" -c gunicorn.conf.py