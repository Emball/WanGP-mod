import gradio as gr

from .refmods import MAX_WEIGHT, PARTS, list_refmod_choices, parse_refmod_spec, serialize_refmod_spec

SLOTS = 8

INFO = ("<B>RefMods are saved reference latents (image, video or audio) that the model attends to like extra reference inputs. "
        "They need no trigger word and stay active while listed, but naming the subject in the prompt helps anchor them. "
        "Weight 0-1 sets detail strength (lower keeps only the coarse look and mood). Above 1 the RefMod is repeated "
        "(2.7 = two full copies plus one at 0.7), which costs extra tokens per copy.</B>")


def _rows_from_spec(text):
    try:
        entries = parse_refmod_spec(text)
    except ValueError as error:
        print(f"[RefMods] selector could not read the saved spec: {error}")
        entries = []
    if len(entries) > SLOTS:
        print(f"[RefMods] selector shows the first {SLOTS} of {len(entries)} saved entries")
    rows = list(entries[:SLOTS])
    return rows + [(None, "all", 1.0)] * (SLOTS - len(rows))


def _spec_to_values(text):
    return [value for row in _rows_from_spec(text) for value in row]


def _values_to_spec(*values):
    return serialize_refmod_spec(zip(values[0::3], values[1::3], values[2::3]))


def _refresh_choices():
    names = list_refmod_choices()
    return [gr.update(choices=names) if index % 3 == 0 else gr.update() for index in range(SLOTS * 3)]


def build_refmods_tab(spec_box):
    choices = list_refmod_choices()
    with gr.Tab("RefMods") as tab:
        gr.Markdown(INFO)
        with gr.Row():
            refresh_button = gr.Button("Refresh List", size="sm")
            clear_button = gr.Button("Clear All", size="sm")
        slots = []
        for index, (name, part, weight) in enumerate(_rows_from_spec(spec_box.value or "")):
            with gr.Row():
                slots.append(gr.Dropdown(choices=choices, value=name, label=f"RefMod {index + 1}", allow_custom_value=True, scale=5))
                slots.append(gr.Dropdown(choices=list(PARTS), value=part, label="Applies To", scale=1, min_width=110))
                slots.append(gr.Slider(minimum=0, maximum=MAX_WEIGHT, step=0.05, value=weight, label="Weight", scale=3, show_reset_button=False))
    for component in slots:
        component.input(fn=_values_to_spec, inputs=slots, outputs=spec_box, show_progress="hidden")
    spec_box.change(fn=_spec_to_values, inputs=spec_box, outputs=slots, show_progress="hidden")
    refresh_button.click(fn=_refresh_choices, outputs=slots, show_progress="hidden")
    tab.select(fn=_refresh_choices, outputs=slots, show_progress="hidden")
    clear_button.click(fn=lambda: "", outputs=spec_box, show_progress="hidden")
    return tab
