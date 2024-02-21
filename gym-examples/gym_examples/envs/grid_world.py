import gym
from gym import spaces
import pygame
import numpy as np

def process_article(article, max_length):
    import re
    
    input_string = article
    pattern = r'([\s\S]*?)\n\n\w+[\s]*\n\n'
    match = re.search(pattern, input_string)

    if match:
        text_before_word = match.group(1)
        output_string = text_before_word
    else:
        output_string = input_string

    output_string = re.sub(r'\s+', ' ', output_string)
    words = re.findall(r"[\w']+|[.,!?;-_=+\(\)\[\]]", output_string)
    return words[:max_length]

def process_title(title):
    import re
    title = re.sub(r'\s+', ' ', title)
    words = re.findall(r"[\w']+|[.,!?;-_=+\(\)\[\]]", title)
    print(words)
    return title

def load_wiki_page():
    import pandas as pd

    wiki = pd.read_csv("/home/gabriel/cours/RL/projet/wikipedia_simple.csv")
    article = wiki.sample()
    return article.to_dict(orient="records")[0]


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, sim_threshold=0.95, max_article_size=1000, max_title_size=100, max_word_length=1000):

        self.max_article_size = max_article_size
        self.sim_threshold = sim_threshold
        self.window_size = 512  # The size of the PyGame window

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


    def _get_obs(self):
        return {
            "title": self._title,
            "words_prox": self._words_prox,
            "words_size": self._words_size,
            "proposed_words": self._proposed_words,
            "fitted_words": self._fitted_words,
            "fitted_title": self._fitted_title,
            }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Load the Wikipedia page and process it
        self._wiki = load_wiki_page() # it's a dictionary with "id", "url", "title", "text"
        self._title = process_title(self._wiki.pop("title"))
        self._article = self._wiki.pop("text")
        self._words = process_article(self._article, self.max_article_size)
        
        self._article_length = len(self._words)
        self._title_length = len(self._title)

        # Initialize the environment corresponding to the loaded Wikipedia page
        self._title = np.zeros(self._title_length, dtype=bool)
        self._words_prox = np.zeros(self._article_length, dtype=float)
        self._words_size = np.array([len(word) for word in self._words], dtype=int)
        self._proposed_words = [] # no proposed words at the beginning
        self._fitted_words = [None] * self._article_length # only gray squares at the beginning
        self._fitted_title = [None] * self._title_length # only gray squares at the beginning

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation
    
    def step(self, action):

        already_proposed = False

        if action not in self._proposed_words:
            self._proposed_words.append(action)

            # Check similarity with title
            for i, word in enumerate(self._title):
                similarity = compute_similarity(word, action)

                # if the similarity is better than the threshold, we fit the true word.
                if similarity>self.sim_threshold:
                    self._fitted_title[i] = word

            # Check similarity with article
            for i, word in enumerate(self._words):
                similarity = compute_similarity(word, action)

                # if the similarity is better than the threshold, we fit the true word
                # and update the proximity to 1.
                if similarity>self.sim_threshold: 
                    self._fitted_words[i] = word
                    self._words_prox[i] = 1

                # if the similarity is better than the previous one, we fit the proposed word
                # and updata the proximity to the new similarity score.
                elif similarity>self._words_prox[i]: 
                    self._fitted_words[i] = action
                    self._words_prox[i] = similarity

        elif action in self._proposed_words:
            already_proposed = True                  


        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

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