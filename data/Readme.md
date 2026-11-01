Latest Scheme Dataset — Data Dictionary and Download Instructions
Version: v1.0 | Date: 2026-10-29
Corresponding scheme: Cosine basis (zero-flux self-consistency) + explicit 
𝐷𝑒𝑓𝑓(𝑥) Deff​(x) + 11-element environmental modulation
Main script: full_spectrum_cos_noflux.py (fixed seed, exactly reproducible)

1. Dataset Overview
Item	Content
Time scales	5: minute / hour / day / month / year
Training set	400 points per scale, 2000 points in total
Test set	100 points per scale, 500 points in total
Number of columns	19 columns (19 in total including the scale label)
Reproduction seeds	Training 61000 + s, test 60000 + s (𝑠s=0..4)
File List
File	Rows	Description
train_scale0_minute.csv … train_scale4_year.csv	400 × 5	Training set, split into separate files by scale
test_scale0_minute.csv … test_scale4_year.csv	100 × 5	Test set, split into separate files by scale
all_data.npz	30 arrays	All data packaged together (x / t_rel / u for each scale)
dataset_manifest.json	—	Shapes, seeds, SHA256 checksums
2. Field Dictionary
Column name	Meaning	Unit	Value range
scale	Time-scale label	—	minute/hour/day/month/year
x	Normalized depth	dimensionless 
[0,1]
[0,1]	0 = soil surface, 1 = deep layer
t_rel	Relative elapsed time (counted from the anchor point)	s	minute 60–3600; year 3.15×10⁷–1.89×10⁸
t_abs	Absolute timestamp	s (unix)	—
T	Temperature	°C	−0.98 – 31.0 (see §4, item 2)
H	Relative humidity	dimensionless	0–1
N,P,K	Macronutrients	dimensionless (normalized)	0–1
Mg,Ca,S	Secondary nutrients	dimensionless (normalized)	0–1
Fe,Mn,Zn,Cu,B	Micronutrients	dimensionless (normalized)	0–1
phi	Porosity	dimensionless volume fraction (m³/m³)	0.45–0.65
u	Reference microbial biomass	dimensionless	−1.74 – 3.74 (may be negative, see §4)
Note on dimensions: phi is a dimensionless volume fraction, not cm. x is normalized depth (dimensionless); converting to physical depth requires multiplying by the total depth (the 30 cm interface is located at 
𝑥=0.6.

3. Environmental Factor Model
Factor	Model	Parameters
Temperature	Ratkowsky (square-root type)	
| multi_pathway_results.npz | Time series of the 18-parameter model for the Pseudomonas case |
