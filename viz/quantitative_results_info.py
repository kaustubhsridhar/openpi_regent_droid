quantitative_results = {
    "move the idli plate to the right": {
        "reaches suitable\ngrasp for\nidli plate": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 50,
            "RICL-π₀-FAST-DROID": 80,
            "π₀-FAST-DROID-finetuned": 60,
            "RICL-π₀-FAST-DROID-finetuned": 90,
            "Diffusion Policy": 20,
        },
        "prev + moves\nto the right": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 20,
            "RICL-π₀-FAST-DROID": 40,
            "π₀-FAST-DROID-finetuned": 50,
            "RICL-π₀-FAST-DROID-finetuned": 80,
            "Diffusion Policy": 0,
        },
    },
    "pick up the poke ball and put it in the tray": {
        "reaches pokeball\nand not distractor": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 40,
            "RICL-π₀-FAST-DROID": 100,
            "π₀-FAST-DROID-finetuned": 100,
            "RICL-π₀-FAST-DROID-finetuned": 100,
            "Diffusion Policy": 10,
        },
        "prev + grasps\npokeball and\nstarts to move": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 10,
            "RICL-π₀-FAST-DROID": 60,
            "π₀-FAST-DROID-finetuned": 30,
            "RICL-π₀-FAST-DROID-finetuned": 70,
            "Diffusion Policy": 10,
        },
        "prev + moves\nto the right": {
            "π₀-FAST-DROID": 0,
            "Retrieve and play": 0,
            "RICL-π₀-FAST-DROID": 40,
            "π₀-FAST-DROID-finetuned": 20,
            "RICL-π₀-FAST-DROID-finetuned": 60,
            "Diffusion Policy": 0,
        },
    },
    "move the squeegee to the right and try to drag it": {
        "reaches the\nsqueegee (not\ndistractor) and\nsuitable grasp": {
            "π₀-FAST-DROID": [10, None],
            "Retrieve and play": [30, None],
            "RICL-π₀-FAST-DROID": [90, None],
            "π₀-FAST-DROID-finetuned": [90, None],
            "RICL-π₀-FAST-DROID-finetuned": [100, None],
            "Diffusion Policy": [50, None],
        },
        "prev + performs\ngrasp and\nstarts to move": {
            "π₀-FAST-DROID": [10, None],
            "Retrieve and play": [10, None],
            "RICL-π₀-FAST-DROID": [70, None],
            "π₀-FAST-DROID-finetuned": [50, None],
            "RICL-π₀-FAST-DROID-finetuned": [90, None],
            "Diffusion Policy": [10, None],
        },
        "prev + moves\nto the right (extra: contact)": {
            "π₀-FAST-DROID": [0, 0],
            "Retrieve and play": [0, 0],
            "RICL-π₀-FAST-DROID": [30, 20],
            "π₀-FAST-DROID-finetuned": [20, 20],
            "RICL-π₀-FAST-DROID-finetuned": [70, 70],
            "Diffusion Policy": [0, None],
        },
    },
    # second round of tasks below
    "pick up the bagel and\nput it in the toaster": {
        "reaches the\nbagel (and\ndoesn't aimlessly\nwander)": {
            "π₀-FAST-DROID": [60, None],
            "RICL-π₀-FAST-DROID": [100, None],
            "π₀-FAST-DROID-finetuned": [100, None],
            "RICL-π₀-FAST-DROID-finetuned": [100, None],
        },
        "prev + performs\ngrasp and\nstarts to move": {
            "π₀-FAST-DROID": [30, None],
            "RICL-π₀-FAST-DROID": [30, None],
            "π₀-FAST-DROID-finetuned": [40, None],
            "RICL-π₀-FAST-DROID-finetuned": [90, None],
        },
        "prev + moves\nbagel close\nto the toaster (extra: completes task)": {
            "π₀-FAST-DROID": [20, 0],
            "RICL-π₀-FAST-DROID": [20, 0],
            "π₀-FAST-DROID-finetuned": [30, 0],
            "RICL-π₀-FAST-DROID-finetuned": [50, 0],
        },
    },
    "push the lever on\nthe toaster": {
        "reaches the\nlever (and\ndoesn't aimlessly\nwander)": {
            "π₀-FAST-DROID": 30,
            "RICL-π₀-FAST-DROID": 40,
            "π₀-FAST-DROID-finetuned": 80,
            "RICL-π₀-FAST-DROID-finetuned": 100,
        },
        "prev + pushes\nthe lever down": {
            "π₀-FAST-DROID": 0,
            "RICL-π₀-FAST-DROID": 20,
            "π₀-FAST-DROID-finetuned": 50,
            "RICL-π₀-FAST-DROID-finetuned": 50,
        },
    },
    "open the door of the bottom shelf": {
        "reaches the\nbottom door": {
            "π₀-FAST-DROID": 0,
            "RICL-π₀-FAST-DROID": 90,
            "π₀-FAST-DROID-finetuned": 100,
            "RICL-π₀-FAST-DROID-finetuned": 100,
        },
        "prev + performs\ngrasp and\nstarts to move": {
            "π₀-FAST-DROID": 0,
            "RICL-π₀-FAST-DROID": 40,
            "π₀-FAST-DROID-finetuned": 50,
            "RICL-π₀-FAST-DROID-finetuned": 100,
        },
        "prev + opens\nthe door": {
            "π₀-FAST-DROID": 0,
            "RICL-π₀-FAST-DROID": 20,
            "π₀-FAST-DROID-finetuned": 20,
            "RICL-π₀-FAST-DROID-finetuned": 60,
        },
    },
    # third round of tasks below (in the setting next to the sink)
    "move the idli plate\nto the sink": {
        "reaches suitable\ngrasp for\nidli plate": {
            "π₀-FAST-DROID": 60,
            "RICL-π₀-FAST-DROID": 80,
        },
        "prev + moves\ninto sink": {
            "π₀-FAST-DROID": 0,
            "RICL-π₀-FAST-DROID": 40,
        },
    },
    "use the squeegee to\nclean the counter": {
        "reaches the\nsqueegee and\nsuitable grasp": {
            "π₀-FAST-DROID": 10,
            "RICL-π₀-FAST-DROID": 90,
        },
        "prev + performs\ngrasp and\nmoves to sink": {
            "π₀-FAST-DROID": 0,
            "RICL-π₀-FAST-DROID": 80,
        },
        "prev + pellets\nfall into sink": {
            "π₀-FAST-DROID": 0,
            "RICL-π₀-FAST-DROID": 40,
        },
    },
    # no-loss-of-capabilities tasks
    "tasks demonstrating no loss-of-capabilities": {
        "move the can\nto the tray": {
            "π₀-FAST-DROID": 80,
            "RICL-π₀-FAST-DROID": 80,
        },
        "pick up the\nmarker and put\nit in the tray": {
            "π₀-FAST-DROID": 80,
            "RICL-π₀-FAST-DROID": 80,
        },
        "place the\napple next\nto the can": {
            "π₀-FAST-DROID": 80,
            "RICL-π₀-FAST-DROID": 80,
        },
    },
}
