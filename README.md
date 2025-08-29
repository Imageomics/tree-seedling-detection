
## Seedling FiftyOne
An introductory approach to using a FiftyOne dataset interface with seedling images using BioCLIP 2 classification attempts is available in `seedling-fo.py`.

To set up:
- Create and activate an environment.
- Install libraries in `seedling-fo_requirements.txt` with e.g.:
```bash
uv pip install -r seedling-fo_requirements.txt
```

To run:
```bash
python seedling-fo.py <path-to-images-dir>
```

Additional flags exist as well.
Currently, you may:
- specify a desired `--confidence` level for open-ended prediction
- use a `--device` (e.g. `cuda`)
- use `--custom-labels` pointint to a CSV with a column of custom labels to predict among
- specify a `--dataset-name` if one has already been executed
- `--force-reprocess` to remove dataset contents with newly processed data

