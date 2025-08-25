# Harnessing Large Language Models for Virtual Reality Exploration Testing: A Case Study

**Authors:**  
Zhenyu Qi<sup>1</sup>, Haotang Li<sup>1</sup>, Hao Qin<sup>2</sup>, Kebin Peng<sup>3</sup>, Sen He<sup>1</sup>, **Xue Qin<sup>4*</sup>**  
<sup>1</sup>Department of Electrical and Computer Engineering, The University of Arizona  
<sup>2</sup>Department of Mathematics, The University of Arizona  
<sup>3</sup>Department of Computer Science, East Carolina University  
<sup>4</sup>Department of Computing Sciences, Villanova University (*Corresponding author)  

📧 Contact: [xue.qin@villanova.edu](mailto:xue.qin@villanova.edu)

---

## 📖 Overview

As the **Virtual Reality (VR)** industry expands, automated **GUI testing** is becoming increasingly critical. **Large Language Models (LLMs)**, with their ability to retain information and analyze both visual and textual data, offer new possibilities for testing and exploration in VR.

This repository accompanies our paper:  
**"Harnessing Large Language Models for Virtual Reality Exploration Testing: A Case Study"**  
We investigate the capabilities of LLMs—particularly **GPT-4o**—for **field of view (FOV) analysis** and **scene understanding** in VR environments.

---

## 🔍 Key Contributions

- **FOV Entity Identification**  
  - LLMs can identify entities in VR field of views.  
  - Prompt engineering improved accuracy from **41.67% → 71.30%**.

- **Entity Feature Description**  
  - Identified entities described with at least **90% accuracy**.  
  - Core features: **color, placement, and shape**.  
  - Combination of features improved entity matching across FOVs (F1 = **0.70**).

- **Scene Recognition & Spatial Understanding**  
  - Structured prompts enabled effective scene-level reasoning in VR.

- **Limitations**  
  - LLMs struggled with **entity labeling**, highlighting future research needs.

---

## 🧪 Case Study Setup

- **Model Used**: GPT-4o  
- **Task**: Automated VR exploration testing via FOV snapshots  
- **Evaluation Metrics**: Accuracy, F1-score  
- **Focus**: Entity detection, description, and scene reasoning  

---

## 📊 Results at a Glance

| Task                                | Performance |
|-------------------------------------|--------------|
| Entity Identification (baseline)    | 41.67%       |
| Entity Identification (prompt-tuned)| 71.30%       |
| Feature Description Accuracy        | ≥ 90%        |
| Entity Matching (F1-score)          | 0.70         |

---

## 📌 Keywords

`Virtual Reality` · `Large Language Models` · `Exploration Testing` · `Automated GUI Testing`

---

## 📄 Citation

If you use this work in your research, please cite our paper:

```bibtex
@misc{qi2025harnessinglargelanguagemodel,
      title={Harnessing Large Language Model for Virtual Reality Exploration Testing: A Case Study}, 
      author={Zhenyu Qi and Haotang Li and Hao Qin and Kebin Peng and Sen He and Xue Qin},
      year={2025},
      eprint={2501.05625},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2501.05625}, 
}
