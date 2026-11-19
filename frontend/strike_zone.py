"""Draw the 13 Statcast regions as a schematic from the catcher's view."""

from html import escape

from frontend.replay import zone_for_catcher, zone_label

ZONE_PATHS = {
    11: "M71 26H216V56H126V146H71Z",
    12: "M216 26H361V146H306V56H216Z",
    13: "M71 146H126V236H216V266H71Z",
    14: "M306 146H361V266H216V236H306Z",
}
for zone in range(1, 10):
    row, column = divmod(zone - 1, 3)
    x, y = 126 + column * 60, 56 + row * 60
    ZONE_PATHS[zone] = f"M{x} {y}h60v60h-60Z"


def chart_description(stand: str, predicted: int | None, actual: int | None) -> str:
    side = {"L": "right", "R": "left"}.get(stand)
    title = "Strike zone from the catcher’s view."
    if side:
        title += f" Batter on the {side}."
    for purpose, zone in (("Predicted", predicted), ("Actual", actual)):
        if zone is not None:
            title += f" {purpose} {zone_label(zone_for_catcher(zone, stand)).lower()}."
    return title


def strike_zone(stand: str, predicted: int | None, actual: int | None) -> str:
    batter_x = 40 if stand == "R" else 392
    regions = "".join(
        f'<path d="{path}" fill="{("#ffffff" if zone < 10 else "#e8eff3")}"/>'
        for zone, path in ZONE_PATHS.items()
    )
    prediction = ""
    if predicted is not None:
        prediction = (
            f'<path class="zone-prediction" d="{ZONE_PATHS[predicted]}" '
            'fill="#b6dce9" stroke="#006d99" stroke-width="2.5"/>'
        )
    outcome = ""
    if actual is not None:
        outcome = (
            f'<path class="zone-actual" d="{ZONE_PATHS[actual]}" '
            'fill="none" stroke="#a44219" stroke-width="3.5"/>'
        )
    title = escape(chart_description(stand, predicted, actual))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 432 304"
      role="img" aria-labelledby="zone-title" stroke-linejoin="round">
      <title id="zone-title">{title}</title>
      <g class="zone-regions" stroke="#627787">{regions}</g>
      {prediction}
      <path d="M126 56H306V236H126Z" fill="none" stroke="#172c43" stroke-width="1.65"/>
      {outcome}
      <path d="M201 273H231V284L216 294L201 284Z" fill="white" stroke="#627787" stroke-width="1.4"/>
      <g fill="#172c43" transform="translate({batter_x} 144)" aria-hidden="true">
        <circle cy="-13" r="6"/>
        <path d="M-9 11V4A9 9 0 0 1 9 4V11Z"/>
        <text y="31" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700">Batter</text>
      </g>
    </svg>
    """
