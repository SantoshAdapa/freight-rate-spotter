\# Freight Rate Prediction Challenge



See `Freight\_Rate\_ML\_Assessment.pdf` for the assessment instructions.



\## What to do



1\. Train and validate your model using `data/train\_test.csv`.

2\. Predict every load in `data/validation.csv`. Each load has a unique `load\_id`.

3\. Fill the matching `predicted\_rate` values in `data/validation\_predictions\_template.csv` and save it as `validation\_predictions.csv`.

4\. Predict every row in `data/december\_chart\_inputs.csv` by filling its `predicted\_rate` column.

5\. Install the scorer requirements and run:



```bash

python -m pip install -r requirements.txt

python score.py --predictions validation\_predictions.csv --december-predictions data/december\_chart\_inputs.csv

```



The scorer validates both files and creates `scorer\_results/candidate\_december.png`.



\## Submit



\- GitHub repository containing your code, dependencies, and run instructions

\- `validation\_predictions.csv`

\- PDF or DOCX report containing your validation, data split approach and `candidate\_december.png`

\- 2-3 minute Loom link

