# Final Assignment: Group Project

For the final assignment you will work on a real ML problem as a group. There is no notebook template this time \- figuring out how to structure and present your work is part of the exercise.

### Group Setup

Groups of **6-7 students**. These are bigger than before because the projects involve more moving parts: collecting or preparing data, generating new data, building models, running experiments, writing up results. Split the work between you.

Pick one of the three projects below. Submit:

1. **First choice** \- the one you want
2. **Second choice** \- your fallback

We want at least one group on each project, so we may shuffle things around after collecting everyone's preferences.

### Hand-in Instructions

IMPORTANT: Name your file `{group_number}_final_assignment.pdf` (e.g., `g03_final_assignment.pdf`).

Hand in a single **PDF report** \- a proper technical report, not a notebook export. Put all your code in an **appendix** at the end. Also submit `.ipynb` or `.py` files, depending on what you build.

### Report Structure

Max **4000 words** in the main report (the appendix does not count). Report template:

1. **Introduction**
2. **Dataset**
3. **Method**
4. **Results**
5. **Discussion**
6. **Conclusion**

The results should be presented in a manner that is easy to interpret (eg: confusion matrices, ROC/AUC metrics or other graphical presentations)

### AI Usage

There are no restrictions on the use of AI tools for this assignment. You are encouraged to use AI as a code assistant. Just be careful: always double-check your analyses, results, and any numbers or claims that come out of AI-generated code. The responsibility for correctness is yours.

### Plan Review

Before diving into the main work, each group must present their plan to the instructors. This should cover what you intend to do, how you plan to split the work, and what your approach will be. You will get feedback and guidance before proceeding. This will be done over Teams, either by chat or in a meeting, depending on what you prefer.

### Expectations

You don't need to solve the whole problem. A well-scoped question with clean methodology and honest analysis will always beat an overambitious project where the evaluation falls apart. Be upfront about what worked and what didn't.

Make use of figures \- confusion matrices, training curves, example predictions, distribution plots. A good figure does more than a paragraph of text.

---

## Projects

### A \- SAM 3 Segmentation and Fine-Tuning

**SAM 3** (Segment Anything Model 3\) is Meta's foundation model for image segmentation. Given an image and a prompt (pointing to a pixel on the image, defining a bounding box in the image, or a text prompt), it produces a segmentation mask without needing to be trained on that specific type of object. It works surprisingly well out of the box on many domains, but its zero-shot performance is not always enough for specialized tasks. That's where fine-tuning comes in.

- Paper: [https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- Project page: [https://ai.meta.com/research/sam3/](https://ai.meta.com/research/sam3/)
- Code: [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)

This project has two parts:

#### 1\. Inference demo on a 5G slice (week 16/17)

Run SAM 3 inference on images or video as part of a 5G network slice setup. The actual 5G slice work is handled as part of the CBU3502 final exercise, not this course. For this part we don't expect any training or tuning, just get the model’s inference engine running and demonstrate segmentation on something relevant for the exercise.

#### 2\. Fine-tuning on a custom dataset

Find a category of objects where SAM 3 struggles to differentiate between similar-looking items (e.g. types of military equipment, specific vehicle models, or similar). Collect your own images of these objects, annotate them, and fine-tune SAM 3\. Make sure you run pretrained SAM 3 on the same held-out images and collect performance data as a baseline, so you can quantify whether fine-tuning actually helps.

#### Goal

Show that SAM 3 inference works in the 5G slice setup, and separately investigate whether fine-tuning on a targeted dataset improves segmentation of objects it does not handle well out of the box. The emphasis in this course is on the ML side \- inference, training, evaluation, and comparison \- not on the 5G setup itself.

---

### B \- Drone Classification from RF Signals

Every drone emits RF signals \- for control, telemetry, and video transmission. These signals have characteristics that differ between drone models and flight modes. The idea behind this project is to pick up those RF emissions and classify what drone is in the air based on the signal alone.

#### Dataset

The **DroneRF dataset** was collected using two RF receivers at different frequency bands, capturing the ambient RF activity while drones were operated under controlled conditions. The dataset contains **227 RF recordings** across several drone types (Bebop, AR, Phantom) and different flight modes (off, on but not flying, hovering, flying, recording video). Each recording is a longer continuous segment \- not a ready-made short sample. You will have to figure out how to turn these raw recordings into something you can feed to an ML model.

- Paper: [https://www.sciencedirect.com/science/article/abs/pii/S0167739X18330760?via%3Dihub](https://www.sciencedirect.com/science/article/abs/pii/S0167739X18330760?via%3Dihub)
- Code: [https://github.com/Al-Sad/DroneRF](https://github.com/Al-Sad/DroneRF)
- Data: [https://data.mendeley.com/datasets/f4c2b4n755/1](https://data.mendeley.com/datasets/f4c2b4n755/1)

#### Goal

Classify drones (or drone activity) from RF signals. Try more than one approach, compare them, and dig into where they disagree or fail.

---

### C \- Malicious Activity Detection from Network Traffic

Every connection that crosses a network leaves a trace: how long it lasted, what protocol it used, how many bytes went back and forth, what service was requested. Buried in those traces are patterns that separate normal traffic from attacks. The catch is that attacks are rare, they come in very different flavors, and a model that just predicts "normal" most of the time can look deceptively accurate. Getting this right means thinking hard about what metrics actually matter and what the model is really learning.

#### Dataset

**UNSW-NB15** was built at the University of New South Wales by running a mix of real normal traffic and synthetic attacks through a controlled network testbed. The result is a dataset of connection records, each described by 49 features (duration, protocol, service, bytes transferred, packet counts, etc.) and labelled as normal or one of nine attack categories: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms. It ships with official train/test CSV splits, so you can get started without heavy data wrangling.

This is a tabular classification problem \- not time-series.

- Paper: [https://ieeexplore.ieee.org/abstract/document/7348942](https://ieeexplore.ieee.org/abstract/document/7348942)
- Data: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

#### Goal

Detect malicious traffic from flow-level features. You decide the formulation (binary \[normal v malignant\]? Multiclass \[normal, analysis, fuzzer, backdoor…ect\]?), which models to use, and how to deal with class imbalance. Show what works, what doesn't, and why.
