## This file is a modified version of the file `grid_world.py` from the gym documentation.
## The tutorial can be found at https://www.gymlibrary.dev/content/environment_creation/.
## The GridWorldEnv class has been modified to replace the grid game by the Pedantle one.

import gym
from gym import spaces
import pygame
import numpy as np
import re
from gym_examples.wrappers.utils import process_article, process_title, load_wiki_page
from gym_examples.wrappers.sim_computer import load_embedding_model, compute_similarity

class PedantleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
            self, 
            test_model=False,
            wiki_file="/home/gabriel/cours/RL/projet/wikipedia_simple.csv",
            render_mode=None, 
            logging=None,
            sim_threshold_true=0.55,
            sim_threshold_fit=0.2, 
            max_article_size=1000, 
            max_title_size=100, 
            max_word_length=1000,
            ):

        self.logging = logging
        self.wiki_file = wiki_file
        self.embedding_model, self.faiss_index = load_embedding_model(test_model, logging)
        self.max_article_size = max_article_size
        self.sim_threshold_true = sim_threshold_true
        self.sim_threshold_fit = sim_threshold_fit
        self.window_size = (1600,900)  # The size of the PyGame window

        # Observations are dictionaries with the title and the words states.
        # - title is a binary vector (0 if not discovered, 1 if discovered).
        # - words_prox is a sequence of floats between 0 and 1, representing 
        #   the proximity of the closest proposed word to the true word. 
        # - words_size is a sequence of integers, representing the size of the true words.
        # - proposed_words is a sequence of strings, representing the proposed words.
        # - fitted_words is a sequence of strings, representing the proposed words that 
        #   best fit some of the article's words.
        # - fitted_title is a sequence of strings, representing the proposed words that
        #   best fit the title.
        self.observation_space = spaces.Dict(
            {
                "index_of_words_to_find": spaces.MultiDiscrete(np.ones(max_article_size), dtype=int),
                "title": spaces.MultiBinary(max_title_size),
                "words_prox": spaces.MultiDiscrete(np.ones(max_article_size), dtype=float),
                "words_size": spaces.MultiDiscrete([max_word_length for i in range(max_article_size)], dtype=int),
                "proposed_words": spaces.Sequence(spaces.Text(max_word_length)),
                "fitted_words": spaces.Sequence(spaces.Text(max_word_length)),
                "fitted_title": spaces.Sequence(spaces.Text(max_word_length)),
            }
        )

        # The action space is only a string of maximum max_action_size.
        self.action_space = spaces.Text(max_word_length)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.logging.info('Environment created with the following parameters:')
        self.logging.info(f'wiki_file: {self.wiki_file}')
        self.logging.info(f'test_model: {test_model}')
        self.logging.info(f'render_mode: {render_mode} \n')

    def _get_obs(self):
        index_of_words_to_find = [i for i, score in enumerate(self._words_prox) if score != 1]
        obs = {
            "index_of_words_to_find": index_of_words_to_find,
            "words_prox": self._words_prox,
            "words_size": self._words_size,
            "proposed_words": self._proposed_words,
            "fitted_words": self._fitted_words,
            "fitted_title": self._fitted_title,
            "title": self._title,
            }
        #self.logging.info(f'The current observation is: {obs} \n')
        return obs
    
    def get_model(self):
        return self.embedding_model, self.faiss_index
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Load the Wikipedia page and process it
        self._wiki = load_wiki_page(self.wiki_file) # it's a dictionary with "id", "url", "title", "text"
        self._title = process_title(self._wiki.pop("title"))
        self._article = self._wiki.pop("text")
        self._words = process_article(self._article, self.max_article_size)
        
        article_length = len(self._words)
        title_length = len(self._title)

        # Initialize the environment corresponding to the loaded Wikipedia page
        self._words_prox = np.zeros(article_length, dtype=float)
        self._words_size = np.array([len(word) for word in self._words], dtype=int)
        self._proposed_words = [] # no proposed words at the beginning
        self._fitted_words = [None] * article_length # only gray squares at the beginning
        self._fitted_title = [None] * title_length # only gray squares at the beginning

        for i, word in enumerate(self._title):
            if not re.match(r'^[a-zA-Z0-9]+$', word):
                self._fitted_title[i] = word
        
        for i, word in enumerate(self._words):
            if not re.match(r'^[a-zA-Z0-9]+$', word):
                self._fitted_words[i] = word
                self._words_prox[i] = 1

        self.logging.info(f'Reset of the environment with the following wikipedia article: {self._wiki} \n')

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}
    
    def step(self, proposed_word):

        already_proposed = False

        if proposed_word not in self._proposed_words:
            self._proposed_words.append(proposed_word)

            # Check similarity with title
            for i, word in enumerate(self._title):
                similarity = compute_similarity(word, proposed_word, self.embedding_model)

                # if the similarity is better than the threshold, we fit the true word.
                if similarity>self.sim_threshold_true:
                    self._fitted_title[i] = word

            # Check similarity with article
            for i, word in enumerate(self._words):
                similarity = compute_similarity(word, proposed_word, self.embedding_model)

                # if the similarity is better than the threshold, we fit the true word
                # and update the proximity to 1.
                if similarity>max(self.sim_threshold_true,self.sim_threshold_fit): 
                    self._fitted_words[i] = word
                    self._words_prox[i] = 1

                # if the similarity is better than the previous one and higher than
                # the fixed threshold of 0.2, we fit the proposed word
                # and update the proximity to the new similarity score.
                elif similarity>self._words_prox[i] and similarity>0.2: 
                    self._fitted_words[i] = proposed_word
                    self._words_prox[i] = similarity

        elif proposed_word in self._proposed_words:
            already_proposed = True                  

        # An episode is done iff the agent has reached the target
        terminated = None not in self._fitted_title
        reward = 1 if terminated else -1  # Binary sparse rewards
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, {}
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(
            self, 
            title_height=40, 
            word_height=20,
            left_margin=20,
            top_margin=20,
            font="lato",
            padding = 5,
            rounded_radius = 5,
            space_between_words = 5,
            space_between_lines = 3,
            ):

        # pygame initialization
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._title_font = pygame.font.Font('freesansbold.ttf',int(title_height*0.9))
            self._word_font = pygame.font.Font('freesansbold.ttf',int(word_height*0.9))
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Draw the background
        background_color = (255, 255, 255)
        canvas = pygame.Surface(self.window_size)
        canvas.fill(background_color)

        # The offset is used to keep track of the position of the next word to be drawn.
        left_offset = left_margin
        top_offset = top_margin

        # Draw the url of the article
        text_color = (0, 0, 200)
        text = self._word_font.render(self._wiki["url"], True, text_color)
        text_rect = text.get_rect()
        text_rect.left = left_offset
        text_rect.top = top_offset
        canvas.blit(text, text_rect)
        top_margin += 2*text_rect.height + 2*space_between_lines
        top_offset = top_margin

        # Draw the last searched words
        max_word_length = 0
        for i, word in enumerate(reversed(self._proposed_words)):
            text_color = (max(255-20*i,0), 0, 0)
            text = self._word_font.render(word, True, text_color)
            text_rect = text.get_rect()
            text_rect.left = left_offset
            text_rect.top = top_offset
            canvas.blit(text, text_rect)
            top_offset += text_rect.height + 2*space_between_lines
            if text_rect.width > max_word_length:
                max_word_length = text_rect.width

        left_margin = 2*left_margin + max_word_length
        left_offset = left_margin
        top_offset = top_margin

        # Draw the title boxes/words
        for i, word in enumerate(self._fitted_title):
            # Process the word (it can be None)
            text_color = (0, 0, 0)
            text = self._title_font.render(word, True, text_color)
            text_rect = text.get_rect()
            text_rect.left = left_offset
            text_rect.top = top_offset
            
            if word is None: # Draw a black square
                # Add padding to the square
                left_offset += 2*padding

                # We compute the width of the square based on the length of the true word
                text_rect.width = title_height//2 * len(self._title[i])
                
                # Draw the square
                rounded_rect = pygame.Rect(text_rect.left - padding, text_rect.top - padding, text_rect.width + 2 * padding, text_rect.height + 2 * padding)
                rounded_rect.center = text_rect.center
                rectangle_color = (0, 0, 0)
                pygame.draw.rect(canvas, rectangle_color, rounded_rect, border_radius=rounded_radius)

                # Update the offset
                left_offset += rounded_rect.width + space_between_words*title_height//word_height

            else:
                # Draw the word
                canvas.blit(text, text_rect)

                # Update the offset
                left_offset += text_rect.width + space_between_words*title_height//word_height

            if left_offset > self.window_size[0] - left_margin:
                left_offset = left_margin
                top_offset += text_rect.height + 2 * padding + space_between_lines

        left_offset = left_margin
        top_offset += text_rect.height + 2*padding + 2*space_between_lines

        # Draw the article boxes/words
        for i, word in enumerate(self._fitted_words):
            sim = self._words_prox[i] # proximity to the true word
            
            if sim <= 0: # Draw a black square
                # Add padding to the square
                left_offset += 2*padding

                # Create a text object only for geometrical purposes
                text_color = (0, 0, 0)
                text = self._word_font.render(word, True, text_color)
                text_rect = text.get_rect() 
                text_rect.left = left_offset
                text_rect.top = top_offset
                # We compute the width of the square based on the length of the true word
                text_rect.width = word_height//2 * len(self._words[i])
                
                # Draw the square
                rounded_rect = pygame.Rect(text_rect.left - padding, text_rect.top - padding, text_rect.width + 2 * padding, text_rect.height + 2 * padding)
                rounded_rect.center = text_rect.center
                rectangle_color = (0, 0, 0)
                pygame.draw.rect(canvas, rectangle_color, rounded_rect, border_radius=rounded_radius)

                # Update the offset
                left_offset += rounded_rect.width + space_between_words

            elif sim < self.sim_threshold_true: # Draw a black square with a grey square behind
                # Add padding to the square
                left_offset += 2*padding

                # Create the text object for the fitted word
                gray = (sim - self.sim_threshold_true) * 255 + 255
                text_color = (gray, gray, gray)
                text = self._word_font.render(word, True, text_color)
                text_rect = text.get_rect() 
                text_rect.left = left_offset
                text_rect.top = top_offset
                
                # Draw the square
                rounded_rect = pygame.Rect(text_rect.left - padding, text_rect.top - padding, text_rect.width + 2 * padding, text_rect.height + 2 * padding)
                rounded_rect.center = text_rect.center
                rectangle_color = (0, 0, 0)
                pygame.draw.rect(canvas, rectangle_color, rounded_rect, border_radius=rounded_radius)

                # Draw the fitted word
                canvas.blit(text, text_rect)

                # Update the offset
                left_offset += rounded_rect.width + space_between_words

            else:
                # Create the text object for the true word
                text_color = (0, 0, 0)
                text = self._word_font.render(word, True, text_color)
                text_rect = text.get_rect() 
                text_rect.left = left_offset
                text_rect.top = top_offset

                # Draw the true word
                canvas.blit(text, text_rect)
                
                # Update the offset
                left_offset += text_rect.width + space_between_words

            if left_offset > self.window_size[0] - left_margin:
                left_offset = left_margin
                top_offset += text_rect.height + 2 * padding + space_between_lines


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()