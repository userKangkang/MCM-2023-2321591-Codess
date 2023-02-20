import numpy as np

# Define the Sustainable Development Goals (SDGs)
sdgs = [
    "No Poverty",
    "Zero Hunger",
    "Good Health and Well-being",
    "Quality Education",
    "Gender Equality",
    "Clean Water and Sanitation",
    "Affordable and Clean Energy",
    "Decent Work and Economic Growth",
    "Industry, Innovation and Infrastructure",
    "Reduced Inequalities",
    "Sustainable Cities and Communities",
    "Responsible Consumption and Production",
    "Climate Action",
    "Life Below Water",
    "Life On Land",
    "Peace, Justice and Strong Institutions",
    "Partnerships for the Goals"
]

# Define the criteria
criteria = [
    "Importance",
    "Achievability",
    "Urgency",
    "Time Horizon",
    "Alignment"
]

# Define the weights of the criteria
weights = np.array([0.25, 0.20, 0.15, 0.20, 0.20])

# Define the performance matrix
performance_matrix = np.array([
    [7, 9, 8, 8, 8, 7, 6, 8, 8, 9, 7, 7, 9, 8, 8, 7, 7],
    [9, 8, 9, 7, 8, 9, 8, 6, 8, 7, 6, 8, 8, 9, 7, 6, 9],
    [8, 8, 7, 9, 9, 8, 9, 7, 7, 9, 8, 8, 8, 7, 9, 9, 7],
    [8, 8, 8, 8, 7, 8, 8, 9, 9, 8, 9, 8, 7, 8, 8, 7, 8],
    [7, 9, 9, 7, 8, 8, 8, 7, 7, 7, 9, 7, 9, 9, 9, 8, 8],
    [7, 6, 7, 8, 8, 8, 6, 7, 7, 8, 8, 7, 7, 7, 6, 7, 7],
    [7, 7, 6, 7, 7, 6, 8, 8, 8, 7, 7, 7, 6, 7, 7, 8, 6],
    [6, 7, 7, 6, 6, 7, 7, 8, 6, 7, 7, 8, 6, 7, 7, 7, 8],
    [7, 6, 7, 7, 7, 8, 8, 7, 7, 6, 8, 7, 7, 7, 6, 7, 7],
    [8, 7, 7, 8, 8, 8, 7, 6, 7, 7, 7, 7, 8, 7, 7, 6, 7],
    [9, 9, 9, 9, 8, 9, 9, 8, 8, 9, 9, 9, 9, 9, 9, 8, 8],
    [8, 8, 8, 8, 9, 9, 9, 9, 8, 7, 9, 8, 8, 8, 9, 9, 8],
    [9, 9, 9, 8, 8, 8, 8, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8],
    [7, 7, 7, 8, 8, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7],
    [7, 6, 7, 7, 7, 7, 8, 7, 7, 6, 7, 7, 7, 7, 6, 7, 7],
    [7, 7, 7, 8, 8, 7, 7, 8, 7, 7, 7, 7, 7, 7, 7, 6, 7],
    [7, 6, 7, 7, 7, 7, 8, 7, 7, 6, 7, 7, 7, 7, 6, 7, 7]
])

# Normalize the performance matrix
n = performance_matrix.shape[0]
normalized_matrix = np.zeros_like(performance_matrix)
for i in range(n):
    col_sum = np.sum(performance_matrix[:, i])
    normalized_matrix[:, i] = performance_matrix[:, i] / col_sum

# Compute the weighted normalized matrix
weighted_matrix = normalized_matrix.T * weights

# Compute the priority vector for each criterion
criteria_priority = np.sum(weighted_matrix, axis=0) / np.sum(weights)

# Compute the priority vector for each SDG
sdg_priority = np.sum(normalized_matrix.T * criteria_priority, axis=1)

# Print the priorities for the SDGs
for i, sdg in enumerate(sdgs):
    print(f"{i + 1}. {sdg}: {sdg_priority[i]:.3f}")

import numpy as np
from ahpy import pairwise_comparisons, Ahp

# Define the pairwise comparisons for the criteria level
economic_social = pairwise_comparisons(
    ["Economic Sustainability", "Social Sustainability"],
    [[1, 3], [1 / 3, 1]]
)

economic_environmental = pairwise_comparisons(
    ["Economic Sustainability", "Environmental Sustainability"],
    [[1, 1 / 5], [5, 1]]
)

social_environmental = pairwise_comparisons(
    ["Social Sustainability", "Environmental Sustainability"],
    [[1, 1 / 7], [7, 1]]
)

# Combine the criteria pairwise comparisons into a matrix
criteria_matrix = np.zeros((3, 3, 3))
criteria_matrix[0, :, :] = economic_social
criteria_matrix[1, :, :] = economic_environmental
criteria_matrix[2, :, :] = social_environmental

# Define the pairwise comparisons for the sub-criteria level
# Note: These comparisons are subjective and can be adjusted based on the decision-maker's opinions
subcriteria_matrix = np.ones((3, 17, 17))
subcriteria_matrix[0, :, :] = 1 / 9  # Economic Sustainability
subcriteria_matrix[1, :, :] = 1 / 7  # Environmental Sustainability
subcriteria_matrix[2, :, :] = 1 / 5  # Social Sustainability

# Define the desirability matrix for the sub-criteria level
# Note: These values are subjective and can be adjusted based on the decision-maker's opinions
desirability_matrix = np.array([
    [0.075, 0.040, 0.050, 0.045, 0.055, 0.060, 0.065, 0.055, 0.040, 0.045, 0.030, 0.035, 0.040, 0.030, 0.025, 0.020,
     0.030],
    [0.010, 0.080, 0.035, 0.020, 0.025, 0.020, 0.025, 0.015, 0.015, 0.020, 0.015, 0.020, 0.015, 0.010, 0.010, 0.010,
     0.015],
    [0.005, 0.030, 0.080, 0.015, 0.020, 0.020, 0.020, 0.010, 0.015, 0.015, 0.010, 0.015, 0.015, 0.010, 0.005, 0.010,
     0.010],
    [0.010, 0.020, 0.025, 0.070, 0.030, 0.025, 0.030, 0.020, 0.015, 0.020, 0.010, 0.015, 0.015, 0.010, 0.010, 0.005,
     0.010],
    [0.015, 0.025, 0.035, 0.030, 0.080, 0.035, 0.045, 0.030, 0.020, 0.020, 0.020, 0.025, 0.015, 0.020, 0.020, 0.015,
     0.025],
    [0.010, 0.015, 0.015, 0.020, 0.025, 0.070, 0.025, 0.020, 0.015, 0.020, 0.015, 0.020, 0.015, 0.010, 0.010, 0.010,
     0.015],
    [0.005, 0.015, 0.015, 0.015, 0.020, 0.025, 0.070, 0.015, 0.015, 0.020, 0.010, 0.015, 0.015, 0.010, 0.010, 0.010,
     0.015],
    [0.010, 0.015, 0.020, 0.015, 0.025, 0.020, 0.025, 0.065, 0.015, 0.020, 0.010, 0.015, 0.015, 0.010, 0.010, 0.005,
     0.015],
    [0.010, 0.015, 0.020, 0.010, 0.020, 0.015, 0.020, 0.015, 0.060, 0.025, 0.010, 0.015, 0.015, 0.010, 0.010, 0.005,
     0.015],
    [0.015, 0.020, 0.020, 0.015, 0.020, 0.020, 0.020, 0.020, 0.025, 0.060, 0.010, 0.015, 0.015, 0.010, 0.010, 0.010,
     0.015],
    [0.005, 0.015, 0.015, 0.010, 0.015, 0.015, 0.020, 0.010, 0.010, 0.010, 0.080, 0.015, 0.010, 0.005, 0.005, 0.005,
     0.010],
    [0.010, 0.015, 0.015, 0.015, 0.025, 0.020, 0.025, 0.015, 0.015, 0.015, 0.015, 0.070, 0.015, 0.010, 0.010, 0.010,
     0.015],
    [0.015, 0.015, 0.020, 0.015, 0.025, 0.015, 0.025, 0.015, 0.015, 0.015, 0.010, 0.015, 0.070, 0.010, 0.010, 0.005,
     0.015],
    [0.015, 0.020, 0.020, 0.020, 0.030, 0.020, 0.025, 0.015, 0.015, 0.020, 0.005, 0.015, 0.015, 0.010, 0.005, 0.010,
     0.015],
    [0.015, 0.015, 0.015, 0.010, 0.015, 0.015, 0.010, 0.010, 0.005, 0.010, 0.005, 0.010, 0.005, 0.080, 0.005, 0.005,
     0.010],
    [0.010, 0.015, 0.015, 0.015, 0.020, 0.020, 0.015, 0.015, 0.010, 0.010, 0.005, 0.010, 0.010, 0.005, 0.080, 0.005,
     0.010],
    [0.010, 0.015, 0.015, 0.015, 0.020, 0.020, 0.015, 0.015, 0.010, 0.010, 0.005, 0.010, 0.005, 0.005, 0.005, 0.075,
     0.010],
    [0.015, 0.015, 0.020, 0.015, 0.025, 0.025, 0.015, 0.015, 0.015, 0.015, 0.010, 0.015, 0.015, 0.010, 0.010, 0.010,
     0.060]])
# Normalize the criteria matrix by columns
criteria_matrix = criteria_matrix / np.sum(criteria_matrix, axis=0)
# Calculate the weighted criteria matrix
weighted_criteria_matrix = np.dot(criteria_matrix, weights)
# Calculate the priority vector of the weighted criteria matrix
priority_vector = np.sum(weighted_criteria_matrix, axis=1) / len(weights)
# Calculate the consistency index (CI)
n = len(weights)
ci = (np.sum(weighted_criteria_matrix, axis=1) - n) / (n - 1)
# Calculate the consistency ratio (CR)
ri_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
ri = ri_dict[n]
cr = ci / ri
print("Priority Vector:")
print(priority_vector)
print("Consistency Index (CI): {:.4f}".format(ci))
print("Consistency Ratio (CR): {:.4f}".format(cr))
# Normalize the performance matrix by columns
performance_matrix = performance_matrix / np.sum(performance_matrix, axis=0)
# Calculate the weighted performance matrix
weighted_performance_matrix = np.dot(performance_matrix, priority_vector)
# Calculate the total score of each goal
total_scores = np.sum(weighted_performance_matrix, axis=1)
# Calculate the ranking of the goals based on their total scores
ranking = np.argsort(total_scores)[::-1]
# Print the results
print("\nGoal Ranking:")
for i, rank in enumerate(ranking): print("{:2d}. Goal {}: {:.4f}".format(i + 1, sdgs[rank], total_scores[rank]))
