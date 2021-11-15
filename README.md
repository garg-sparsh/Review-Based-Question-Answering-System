# Review-Based QA System
------ 

Can you get the answers of specifc questions like “does this guitar come with a strap?” or “can I use this camera at night?” when you want to buy these products online?

It may be hard to answer from the product description alone.

One alternative is to post your question on the community QA platforms provided by websites like Amazon, but it usually takes days to get an answer (if at all).

It would be nice if we get an immediate answer like the Google Example below

![image.png](attachment:image.png)

## Objective
------

To predict the answers of the question asked by the users by using the reviews of those products.

## The Dataset
--------

To build our QA system, we'll use SubjQA dataset which has 10K customer reviews in English about products and services in the areas of TripAdvisor, Restaurants, Movies, Books, Electronics, and Grocery. 

Firstly we'll focus on bulding the QA for the Electronics Domain, Lets' download the data from [Hugging Face Hub](https://huggingface.co/datasets/subjqa)


```python
from datasets import load_dataset
import pandas as pd

subjqa = load_dataset("subjqa", "electronics")
subjqa.set_format("pandas")
# Flatten the nested dataset columns for easy access
dfs = {split:ds[:] for split, ds in subjqa.flatten().items()}

for split, df in dfs.items():
    print(f"Number of questions in {split}: {df['id'].nunique()}")
```

    Reusing dataset subjqa (/home/ma/sparsh/.cache/huggingface/datasets/subjqa/electronics/1.1.0/e5588f9298ff2d70686a00cc377e4bdccf4e32287459e3c6baf2dc5ab57fe7fd)



      0%|          | 0/3 [00:00<?, ?it/s]


    Number of questions in train: 1295
    Number of questions in test: 358
    Number of questions in validation: 255


We can notice that dataset is quite small with just 1908 examples, it cna simulate the real world scenario where we don't get much data as it quite labor-intensive and time consuming.

### Data Description

![Screen%20Shot%202021-11-14%20at%204.31.06%20PM.png](attachment:Screen%20Shot%202021-11-14%20at%204.31.06%20PM.png)

Lets check few examples of the Data


```python
qa_cols = ["title", "question", "answers.text",
           "answers.answer_start", "context"]
sample_df = dfs["train"][qa_cols].sample(5, random_state=7)
sample_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>question</th>
      <th>answers.text</th>
      <th>answers.answer_start</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>791</th>
      <td>B005DKZTMG</td>
      <td>Does the keyboard lightweight?</td>
      <td>[this keyboard is compact]</td>
      <td>[215]</td>
      <td>I really like this keyboard.  I give it 4 star...</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>B00AAIPT76</td>
      <td>How is the battery?</td>
      <td>[]</td>
      <td>[]</td>
      <td>I bought this after the first spare gopro batt...</td>
    </tr>
    <tr>
      <th>961</th>
      <td>B0074BW614</td>
      <td>How is the cell phone screen?</td>
      <td>[The interface takes a few tries to get used t...</td>
      <td>[535]</td>
      <td>Don't get me wrong, I love my e-ink kindle to ...</td>
    </tr>
    <tr>
      <th>1188</th>
      <td>B00BGGDVOO</td>
      <td>Do you have any computer with mouse?</td>
      <td>[]</td>
      <td>[]</td>
      <td>After deciding to ditch cable TV I started to ...</td>
    </tr>
    <tr>
      <th>999</th>
      <td>B007P4VOWC</td>
      <td>How is the camera?</td>
      <td>[]</td>
      <td>[]</td>
      <td>I purchased the Tab 2 for my fianc&amp;eacute; and...</td>
    </tr>
  </tbody>
</table>
</div>



From above examples we can make some observations:
- In answers.text are empty wehere labeler cant find the answers of the asked questions from the review context
- We can use the answers.answer_start to trace the answer from the context.

Let's check what type of questions are in the training set


```python
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, show  
from bokeh.resources import INLINE
output_notebook(resources=INLINE)
counts = {}
question_types = ["What", "How", "Is", "Does", "Do", "Was", "Where", "Why"]

for q in question_types:
    counts[q] = dfs["train"]["question"].str.startswith(q).value_counts()[True]
print(counts)
from bokeh.plotting import figure, output_file, show
xvals = list(counts.keys())
yvals = list(counts.values())
fig = figure(x_range = xvals, plot_width = 400, plot_height = 300)
#cols = ['navy','cyan','orange']
fig.vbar(x = xvals, top = yvals, width = 0.5)
show(fig)
```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="3919">Loading BokehJS ...</span>
</div>




    {'What': 236, 'How': 780, 'Is': 100, 'Does': 45, 'Do': 83, 'Was': 12, 'Where': 28, 'Why': 21}









<div class="bk-root" id="3be61689-64a4-4994-bfc0-d21d1f77d717" data-root-id="3920"></div>





We can observe that questions beginning with "How", "What", and "Is" are the most asked question, others are few in numbers. Let's check how these questions look like?


```python
for question_type in ["How", "What", "Is"]:
    for question in dfs['train'].query("question.str.contains('%s')"%question_type, engine='python').sample(n=3, random_state=42)['question']:
        print(question)
    print()
```

    How is the camera?
    How do you like the control?
    How fast is the charger?
    
    What is direction?
    What is the quality of the construction of the bag?
    What is your impression of the product?
    
    Is this how zoom works?
    Is sound clear?
    Is it a wireless keyboard?
    



```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
(dfs["train"].groupby("title")["review_id"].nunique().hist(bins=50, alpha=0.5, grid=False, ax=ax))
plt.xlabel("Number of reviews per product")
plt.ylabel("Count");
```


    
![png](output_8_0.png)
    

