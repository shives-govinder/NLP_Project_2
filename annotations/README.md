# Gold set for the normaliser

`gold_normalisation.csv` holds about 300 historical isiXhosa words with their modern spelling.
It is used to tune the normaliser (rows with `split = validation`) and to report its accuracy
(rows with `split = test`). Never tune on test rows.

## How to build it

1. Run `python -m nlp2.normalise candidates`. This writes `gold_candidates_to_annotate.csv`:
   300 word types sampled evenly from rare, mid-frequency and frequent words, each with one
   context sentence. No suggested answer is pre-filled, so the annotator is not anchored by
   our own normaliser.
2. For each row, fill in `modern` with the word as it would be spelled in modern standard
   isiXhosa. If the spelling would not change, copy the historical form. If the word is an
   OCR error, a name or not isiXhosa, write the historical form and explain in `notes`.
3. Fill in `annotator` (initials). Ideally a first-language isiXhosa speaker checks every row.
   Credit them in the acknowledgements, with their permission.
4. Save the completed file as `gold_normalisation.csv` (keep the column order).
