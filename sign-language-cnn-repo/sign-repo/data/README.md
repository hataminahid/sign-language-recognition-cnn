# Dataset (not included)

The American Sign Language letter images were provided with the course
assignment (`docs/assignment3_spec.pdf`). File names follow a Roboflow-style
pattern (`W24_jpg.rf.<hash>.jpg`). Place the data like this
(or point `SIGN_DATA_DIR` to it):

```
data/SignLanguageData/
├── train/   *.jpg  +  _annotations.csv     (1512 images)
├── valid/   *.jpg  +  _annotations.csv     (144 images)
└── test/    *.jpg  +  _annotations.csv     (72 images)
```

`_annotations.csv` has **no header row**; columns are
`image_name, xmin, ymin, xmax, ymax, label` (pixel coordinates of the hand box,
label = letter A–Z).

The notebook writes `cropped/`, `resized_64|128|224/` and `normalized_128/`
sub-folders **inside** each split folder; they are git-ignored.
