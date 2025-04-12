quantitative_results = {
    "move the idli plate to the right": {
        "reaches suitable grasp for idli plate": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 50,
            "Regentic-π₀-FAST-DROID": 80,
            "π₀-FAST-DROID finetuned on 20 demos": 60,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 90,
        },
        "prev + moves to the right": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 20,
            "Regentic-π₀-FAST-DROID": 40,
            "π₀-FAST-DROID finetuned on 20 demos": 50,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 80,
        },
    },
    "pick up the poke ball and put it in the tray": {
        "reaches pokeball and not distractor": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 40,
            "Regentic-π₀-FAST-DROID": 100,
            "π₀-FAST-DROID finetuned on 20 demos": 100,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 100,
        },
        "prev + grasps pokeball and starts to move": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 10,
            "Regentic-π₀-FAST-DROID": 60,
            "π₀-FAST-DROID finetuned on 20 demos": 30,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 70,
        },
        "prev + moves to the right": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 0,
            "Regentic-π₀-FAST-DROID": 40,
            "π₀-FAST-DROID finetuned on 20 demos": 20,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 60,
        },
    },
    "move the squeegee to the right and try to drag it": {
        "reaches the squeegee (not distractor) and suitable grasp": {
            "π₀-FAST-DROID": [10, None],
            "Retrieve and play": [30, None],
            "Regentic-π₀-FAST-DROID": [90, None],
            "π₀-FAST-DROID finetuned on 20 demos": [90, None],
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": [100, None],
        },
        "prev + performs grasp and starts to move": {
            "π₀-FAST-DROID": [10, None],
            "Retrieve and play": [10, None],
            "Regentic-π₀-FAST-DROID": [70, None],
            "π₀-FAST-DROID finetuned on 20 demos": [50, None],
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": [90, None],
        },
        "prev + moves to the right (extra: contact)": {
            "π₀-FAST-DROID": [0, 0],
            "Retrieve and play": [0, 0],
            "Regentic-π₀-FAST-DROID": [30, 20],
            "π₀-FAST-DROID finetuned on 20 demos": [20, 20],
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": [70, 70],
        },
    },
    "pick up the bagel and put it in the toaster": {
        "reaches the bagel (and doesn't aimlessly wander)": {
            "π₀-FAST-DROID": [60, None],
            "Regentic-π₀-FAST-DROID": [100, None],
            "π₀-FAST-DROID finetuned on 20 demos": [100, None],
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": [100, None],
        },
        "prev + performs grasp and starts to move": {
            "π₀-FAST-DROID": [30, None],
            "Regentic-π₀-FAST-DROID": [30, None],
            "π₀-FAST-DROID finetuned on 20 demos": [30, None],
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": [90, None],
        },
        "prev + moves bagel close to the toaster (extra: complete task)": {
            "π₀-FAST-DROID": [20, 0],
            "Regentic-π₀-FAST-DROID": [20, 0],
            "π₀-FAST-DROID finetuned on 20 demos": [10, 0],
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": [50, 0],
        },
    },
    "press the lever on the toaster": {
        "reaches the lever (and doesn't aimlessly wander)": {
            "π₀-FAST-DROID": 30,
            "Regentic-π₀-FAST-DROID": 40,
            "π₀-FAST-DROID finetuned on 20 demos": 80,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 100,
        },
        "prev + presses the lever down": {
            "π₀-FAST-DROID": 0,
            "Regentic-π₀-FAST-DROID": 20,
            "π₀-FAST-DROID finetuned on 20 demos": 50,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 50,
        },
    },
    "open the door of the bottom shelf": {
        "reaches the bottom door": {
            "π₀-FAST-DROID": 0,
            "Regentic-π₀-FAST-DROID": 90,
            "π₀-FAST-DROID finetuned on 20 demos": 100,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 100,
        },
        "prev + performs grasp and starts to move": {
            "π₀-FAST-DROID": 0,
            "Regentic-π₀-FAST-DROID": 40,
            "π₀-FAST-DROID finetuned on 20 demos": 50,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 100,
        },
        "prev + opens the door": {
            "π₀-FAST-DROID": 0,
            "Regentic-π₀-FAST-DROID": 10,
            "π₀-FAST-DROID finetuned on 20 demos": 20,
            "Regentic-π₀-FAST-DROID Regentic-tuned on 20 demos": 60,
        },
    },
}
