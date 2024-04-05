import json
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import matplotlib.animation as animation


RESULT_FILE = "results/results_v3_29_03_08_20.json"
ACTIONS = {
    "Choose a word from a predefined list": 1,
    "Guess the first grey word": 2,
    "Guess a random word": 3,
    "Guess the closest word of the last targetted word": 4,
    "Propose the word that is the most likely to fit a grey word": 5,
    "Guess the grey word that fits the best the article": 6,
    "Propose a word close to one of the revealed words of the title": 7,
}

# Load the JSON file
with open(RESULT_FILE) as file:
    data = json.load(file)

q_values_list = data['q']
nb_words = data['nb_words']
state_visits = data['state_visits']

# Define the Q-values and the actions labels
col_labels = list(ACTIONS.keys())
wrapped_labels = [textwrap.fill(label, 10) for label in col_labels]

# Compute the best actions for each state according to the q-values
best_actions_list = np.zeros_like(q_values_list)
for episode, q_values in enumerate(q_values_list):
    q_values = np.array(q_values)
    best_actions_id = q_values.argmax(axis=1)
    for state, id in enumerate(best_actions_id):
        if q_values[state,id] > 0:
            best_actions_list[episode, state, id] = 1

# Create a new figure for each plot
plt.figure(figsize=(20, 15))

# Plotting the first graph
plt.plot(nb_words)
plt.fill_between(range(len(nb_words)), nb_words, alpha=0.5)
plt.title('Number of words before success or abandon', fontsize=20)
plt.xlabel('Episode')
plt.ylabel('Number of proposed words')
plt.axhline(y=600, color='red', linestyle='dotted', label='Give up threshold')
plt.legend()
plt.show()

# Create a new figure for each plot
plt.figure(figsize=(20, 15))

# Plotting the second graph
plt.plot(state_visits)
plt.fill_between(range(len(state_visits)), state_visits, alpha=0.5)
plt.title('Visited states', fontsize=20)
plt.xlabel('State')
plt.ylabel('Number of visits')
plt.show()

# Create a new figure for each plot
plt.figure(figsize=(20, 15))

# Plotting the Q-values
im = plt.imshow(q_values[:100], aspect='auto')
plt.colorbar(im)
plt.xticks(ticks=range(len(col_labels)), labels=wrapped_labels)
plt.gca().xaxis.set_ticks_position('top') 
plt.xlabel('Action')
plt.ylabel('State')
plt.title('Q-values', fontsize=20)
plt.show()

# Create a new figure for each plot
plt.figure(figsize=(20, 15))

# Plotting best actions for each state
im = plt.imshow(best_actions_list[-1][:300], aspect='auto')
plt.colorbar(im)
plt.xticks(ticks=range(len(col_labels)), labels=wrapped_labels)
plt.gca().xaxis.set_ticks_position('top') 
plt.xlabel('Action')
plt.ylabel('State')
plt.title('Best actions for each state', fontsize=20)
plt.show()

# Create a new figure for the animation
fig, axs = plt.subplots(figsize=(20, 15))

# Define the update function for the animation
def update(frame):
    axs.clear()
    im = axs.imshow(best_actions_list[frame*10][:60], aspect='auto')
    axs.set_xticks(ticks=range(len(col_labels)))
    axs.set_xticklabels(wrapped_labels)
    axs.xaxis.set_ticks_position('top') 
    axs.set_xlabel('Action')
    axs.set_ylabel('State')
    axs.set_title('Q-values (Frame: {})'.format(frame*10), fontsize=20)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(q_values_list)//10, interval=1)

# Display the animation
plt.show()
