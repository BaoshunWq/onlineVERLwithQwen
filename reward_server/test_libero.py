import requests

URL = "http://127.0.0.1:45678/score"

payload = {
    "responses": [
        {"outputs":[{"text":"place the red bowl onto the left shelf"}]},
    ],
    "metas": [
        {
            "original_instruction": "put the red bowl on the left shelf",
            "suite": "libero_object",
            "task_id": 3,
            "seed": 0
        },
    ],
    "reward_function_kwargs": {
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.5,
        "num_trials_per_task": 1,
        "center_crop": 1,
        "libero_cfg": {
            "model_family": "openvla",
        }
    }
}

resp = requests.post(URL, json=payload, timeout=1800)
print(resp.status_code)
print(resp.json())
