# ===============================================================================================================================
# This is the raw code from the cell that produced the nice confusion matrices I used in my final report of my undergrad project.
# ===============================================================================================================================

# Look how accurate the by particle predictions were

cm = confusion_matrix(y_test, preds)

particle_labels = [r'$!\mu$', r'$\gamma$', r'$e$', r'$\mu$']
def eff(m, n):
    return m/n

def uncertainty(m, n):
    return np.sqrt(eff(m, n) * (1 - eff(m, n))/n)

# Create efficiency and uncertainty arrays
eff_matrix = np.zeros_like(cm, dtype=float)
uncertainty_matrix = np.zeros_like(cm, dtype=float)

# Fill the matrices with efficiency and uncertainty
for i in range(len(particle_labels)):
    for j in range(len(particle_labels)):
        total = np.sum(cm[i])  # Total for the true class (row sum)
        true_positive = cm[i, j]  # True positives (diagonal element)
        
        if total > 0:
            eff_matrix[i, j] = eff(true_positive, total)
            uncertainty_matrix[i, j] = uncertainty(true_positive, total)

# Create formatted labels that include efficiency + uncertainty and tally
formatted_values = []
for i in range(len(particle_labels)):
    row = []
    for j in range(len(particle_labels)):
        eff_val = r'$\mathbf{' + f'{eff_matrix[i, j]*100:.2g}' + r'\%}$'
        unc_val = r'$\mathbf{' + f'{uncertainty_matrix[i, j]*100:.2g}' + r'\%}$'
        tally = f'({cm[i, j]})'  # Raw count
        
        # Combine efficiency, uncertainty, and tally
        formatted_string = f'{eff_val} $\pm$ {unc_val}\n\n{tally}'
        row.append(formatted_string)
    formatted_values.append(row)

# Plotting the confusion matrix with efficiency, uncertainty, and tally
plt.figure(figsize=(8*(4/3), 6))

sns.heatmap(cm, annot=formatted_values, fmt='', cmap='Reds', 
            xticklabels=particle_labels, yticklabels=particle_labels)

plt.xlabel('------------------------------------ Predicted Labels ------------------------------------')
plt.ylabel('-------------------------- True Labels --------------------------')
plt.show()

print(eff(np.trace(cm), np.sum(cm))*100)
print(uncertainty(np.trace(cm), np.sum(cm))*100)