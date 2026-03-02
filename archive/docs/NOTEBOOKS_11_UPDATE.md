# Notebooks 11.1 and 11.2 Update: PMI Density Plots Added

**Date:** 2025-01-14  
**Update:** Added PMI distribution density histograms to both notebooks

## Summary

Successfully added PMI density plots (similar to notebook 11) to both notebooks 11.1 and 11.2.

## Changes Made

### **Both Notebooks:**
- **New Cell 10:** Markdown header "## PMI Distribution Density Plot"
- **New Cell 11:** Code generating PMI histogram with density normalization

### **Notebook 11.1 (OLD Summation Method):**
- **Color:** Red
- **Title:** "PMI Distribution (OLD Summation Method)"
- **Label:** "Hetionet (OLD Summation Method)"
- **Output:** `old_method_pmi_distribution.png`

### **Notebook 11.2 (CORRECTED Option A Method):**
- **Color:** Blue
- **Title:** "PMI Distribution (Option A Method)"
- **Label:** "Hetionet (Option A Method)"
- **Output:** `option_a_pmi_distribution.png`

## Notebook Structure (Updated)

Both notebooks now have 17 cells (increased from 15):

```
Cell 0:  Header (markdown)
Cell 1:  Imports (code)
Cell 2:  Configuration header (markdown)
Cell 3:  Configuration (code)
Cell 4:  Helper Functions header (markdown)
Cell 5:  Helper functions (code)
Cell 6:  Function description header (markdown)
Cell 7:  Compute function (code) - OLD vs Option A
Cell 8:  Analysis header (markdown)
Cell 9:  Run analysis (code)
Cell 10: PMI Distribution header (markdown) **NEW**
Cell 11: PMI density plot (code) **NEW**
Cell 12: Scatter plots header (markdown)
Cell 13: Degree-stratified scatter plots (code)
Cell 14: Save results header (markdown)
Cell 15: Save results (code)
Cell 16: Conclusions (markdown)
```

## Visualization Details

The PMI density histogram shows:

1. **Distribution shape:** Normalized histogram (density=True, area = 1.0)
2. **Bins:** 50 bins for smooth distribution
3. **Mean line:** Vertical dashed line showing mean PMI
4. **Statistics:** Printed below plot (mean, median, std)
5. **Styling:** Professional appearance matching notebook 11

## Expected Output Statistics

**Notebook 11.1 (OLD method):**
- Mean PMI: ~7.11 (same as notebook 11)
- Shows full PMI distribution from old summation method

**Notebook 11.2 (CORRECTED method):**
- Mean PMI: ~7.11 (same as notebook 11)
- Shows full PMI distribution from Option A method

**Note:** Both methods produce similar PMI distributions because PMI is calculated from observed vs predicted frequencies. The key difference is that 11.1 has invalid probabilities > 1.0 (visible in scatter plots), but PMI can still be calculated.

## Comparison with Notebook 11

The density plot matches the format from notebook 11 (cell 17, axes[0,0]):
- Same bins (50)
- Same density normalization
- Same mean line styling
- Same professional appearance

**Difference:** Notebooks 11.1 and 11.2 show only Hetionet (no null comparison), since they analyze a single network for demonstration.

## Validation

- [x] Both notebooks have valid Python syntax
- [x] Total cells: 17 each
- [x] Code cells: 8 each
- [x] PMI plots inserted at correct position (after analysis, before scatter plots)
- [x] Proper JSON formatting maintained
- [x] No emojis
- [x] Professional styling

## Usage

When you run the notebooks, they will now generate 3 plots each:

**Notebook 11.1:**
1. `old_method_pmi_distribution.png` - PMI density histogram
2. `old_method_degree_scatter_plots.png` - 3x2 scatter grid

**Notebook 11.2:**
1. `option_a_pmi_distribution.png` - PMI density histogram
2. `option_a_degree_scatter_plots.png` - 3x2 scatter grid

## Files Modified

- `notebooks/11.1_empirical_vs_analytical_compositional.ipynb`
- `notebooks/11.2_empirical_vs_analytical_compositional.ipynb`

No other files changed.

## Next Steps

The notebooks are ready to run. The PMI density plots provide:
1. Quick visual comparison with notebook 11
2. Validation that results align with full analysis
3. Publication-ready histogram for documentation
4. Consistent visualization across methodological comparison notebooks

