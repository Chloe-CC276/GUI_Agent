from src.executor.keyboard import KeyboardController


keyboard = KeyboardController(
    pause=5.0,
    fail_safe=True,
    default_interval=0.03,
    dry_run=False,
)

print(keyboard)

keyboard.write("Hello GUI Agent")
keyboard.enter()

keyboard.hotkey("ctrl", "a")
keyboard.copy()
keyboard.paste()

keyboard.press("down", presses=3)
keyboard.escape()