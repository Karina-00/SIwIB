from gymnasium import Env
import numpy as np
from collections import deque
import random
import math
from gymnasium import error, spaces, utils


# one episode is one day

class Patient(Env):

    def __init__(self, behaviour_threshold=25, has_family=True,
                 good_time=1, habituation=False, time_preference_update_step = 100000000,
                 patient_profile=0, compete_threshold=0.1):
        self.patient_profile = patient_profile
        self.compete_threshold = compete_threshold
        self.record = -1
        self.record_broken = False
        self.behaviour_threshold = behaviour_threshold
        self.has_family = has_family
        self.good_time = good_time # 0 morning, 1 midday, 2 evening, 3 night
        self.habituation = habituation
        self.time_preference_update_step = time_preference_update_step
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete([2, 3, 2,
                                                       2, 2, 2, 2,
                                                       2, 2,
                                                       4, 2,
                                                       24, 24])
        self.activity_p = 0
        self.activity_s = 0
        self.hour_steps = 0
        self.env_steps = 0
        self.max_notification_tolerated = 3
        self.confidence_threshold = 4
        self.week_days = deque(np.arange(1, 8), maxlen=7)
        self.hours = deque(np.arange(0, 24), maxlen=24)
        self.rr = []

        self.valence_list = random.choices([0, 1], weights=(0.9, 0.1), k=23)
        self.arousal_list = random.choices([0, 1, 2], weights=(0.4, 0.2, 0.4), k=23)
        self.activity_performed = [0]
        self.num_performed = []
        self.num_notified = []
        self._start_time_randomiser()
        self.time_of_the_day = self.hours[0]
        self.day_of_the_week = self.week_days[0]  # 1 Monday, 7 Sunday
        self.motion_activity_list = random.choices(['stationary', 'walking'], weights=(1.0, 0.0), k=24) # in last 24 hours
        self.awake_list = random.choices(['sleeping', 'awake'], weights=(0.15, 0.85), k=24)# insomnia
        self.last_activity_score = np.random.randint(0, 2)  # 0 negative, 1 positive
        self.location = 'home' if 1 < self.time_of_the_day < 7 else np.random.choice(['home', 'other'])
        self._update_emotional_state()
        self._initialise_awake_probailities()
        self.h_slept = []
        self.h_positive = []
        self.h_nonstationary = []
        self.observation_list = [self._get_current_state()]
        self.reset()
        # valence =1 #negative/ positive
        # arousal = 1 # low, mid, high
        # cognitive_load = 0 # low, high

        # sleeping=0 # no/yes
        # number_of_hours_slept, more 1 , 0 less than7
        # last_activity_score =1 # low/ high percived benefit
        # time_since_last_activity = 1 # less 0 or 1 more than 24 hours

        # location =0 # home/ other
        # motion_activity = 0 #stationary, walking

        # day_time = 0 # morning 6am-10am, midday 10am-6pm, evening 6pm-10pm, night 10pm-6am
        # week_day  = 1 # week day, weekend

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.activity_p = 0
        self.activity_s = 0
        self.hour_steps = 0
        return self._get_current_state(), self._get_current_info(action=0)

    def update_after_day(self):
        if self.activity_s != 0:
            self.rr.append(self.activity_p / self.activity_s)
        else: # unsucessful training
            self.rr.append(np.nan)
            # self.num_notified.append(np.nan)
        self.num_notified.append(self.activity_s)
        self.num_performed.append(self.activity_p)
        self.h_slept.append(self.awake_list[-24:].count('sleeping'))
        self.h_positive.append(sum(self.valence_list[-24:]))
        self.h_nonstationary.append(self.motion_activity_list[-24:].count('walking'))
        if self.motion_activity_list[-24:].count('walking') > self.record:
            self.record = self.motion_activity_list[-24:].count('walking')
            self.record_broken = True
        self.reset()
        if self.habituation:
            self.behaviour_threshold = self.behaviour_threshold + 0.15 # increase the threshold for performing action

    def step(self, action: tuple):
        info = self._get_current_info(action)

        if action == 1:
            self.activity_s = self.activity_s + 1
            behaviour = self.fogg_behaviour(info['motivation'], info['ability'], info['trigger'])
            self.observation_list.append(self._get_current_state())
            if behaviour:
                self.activity_p = self.activity_p + 1
                self.activity_performed.append(1)
                self._update_patients_activity_score()
                reward = 20
            else:
                self.activity_performed.append(0)
                if self.activity_s < self.max_notification_tolerated:
                    reward = -1
                else:
                    reward = -10
        else:
            reward = 0.0
        # info['reward'] = reward

        self.update_state()
        self.hour_steps = self.hour_steps + 1
        self.env_steps = self.env_steps + 1
        if self.hour_steps == 24:
            self.update_after_day()
            done = True
        else:
            done = False
        if self.env_steps > self.time_preference_update_step:
            self.good_time = 2 # update time preference to be in the evening
        state = self._get_current_state() 
        return state, reward, done, False, info


    # Work-around for Gymnasium
    def _get_current_info(self, action):
        info = dict()
        info['motivation'] = self.get_motivation()
        info['ability'] = self.get_ability()
        info['trigger'] = self.get_trigger()
        info['action'] = action
        return info


    def _get_current_state(self):
        # valence =1 #negative/ positive
        # arousal = 1 # low, mid, high
        # cognitive_load = 0 # low, high

        # sleeping=0 # no/yes
        # number_of_hours_slept in last 24h sufficient 7
        # last_activity_score =1 # low/ high percived benefit
        # time_since_last_activity = 1 # less 0 or 1 more than 24 hours

        # location =0 # home/ other
        # motion_activity = 0 #stationary, walking

        # day_time = 0 # morning 6am-10am, midday 10am-6pm, evening 6pm-10pm, night 10pm-6am
        # week_day  = 1 # week day, weekend

        location = 1 if self.location == 'home' else 0
        sleeping = 1 if self.awake_list[-1] == 'sleeping' else 0
        d = dict([(y, x) for x, y in enumerate(sorted({'stationary', 'walking'}))])
        week_day = self._get_week_day()
        day_time = self._get_time_day()
        t = self._time_since_last_activity()
        number_of_hours_slept = 1 if self.awake_list[-24:].count('sleeping') >= 7 else 0

        obs = np.array([self.valence, self.arousal, self.cognitive_load,
                         sleeping, number_of_hours_slept, self.last_activity_score, t,
                         location, d[self.motion_activity_list[-1]],
                         day_time, week_day, self.activity_s, self.activity_p])
        return obs

    def _time_since_last_activity(self):
        if self.activity_p == 0:
            return 1  # more than 24 hours
        else:
            return 0

    def fogg_behaviour(self, motivation: int, ability: int, trigger: bool) -> bool:
        """"
        Function that decides if the behaviour will be performed or not based on Fogg's Behavioural Model
        """
        behaviour = motivation * ability * trigger
        behaviour = self._consider_stress_and_sleep(behaviour)
        return behaviour > self.behaviour_threshold
    
    def _consider_stress_and_sleep(self, behaviour: int):
        if self.patient_profile in (1, 2):
            if self.valence == 0:
                stress_decrease = random.uniform(0, 0.5)
                behaviour = behaviour * stress_decrease
        if self.patient_profile in (1, 2, 3):
            number_of_hours_slept = 1 if self.awake_list[-24:].count('sleeping') >= 7 else 0
            if number_of_hours_slept == 0:
                sleep_decrease = random.uniform(0, 0.5)
                behaviour = behaviour * sleep_decrease
        return behaviour
    
    def get_motivation(self):
        """
        Factors impacting patient's motivation:
        1) Jowsey et al (2014) "What motivates Australian health service users with chronic illness to engage in
        self-management behaviour?"

         - internal factors:  valence positive(+), negative (-)
            "Remaining positive was one of the most important strategies many used for optimizing and
             controlling their health"
         - external factors: family (+), no family (-)
         - demotivators: high past activity score (+),  low (-)
            "perceiving self-management behaviour as having limited benefit"

        2) Dolsen et al (2017) "Sleep the night before and after a treatment session: A critical ingredient for
        treatment adherence?"
        Axelsson et al (2020) V

        "sleepiness may be a central mechanism by which impaired alertness, for example, due to insufficient sleep,
        contributes to poor quality of life and adverse health. We propose that sleepiness helps organize behaviors
         toward the specific goal of assuring sufficient sleep, in competition with other needs and incentives" Axelsson

         - hours of sleep the previous night, sufficient(+), insufficient(-)

         agency and motivation (MHealth)

        """
        number_of_hours_slept = self.awake_list[-24:].count('sleeping')
        sufficient_sleep = 1 if number_of_hours_slept > 7 else 0
        increased_motivation = self._increased_motivation()

        return self.valence + self.has_family + self.last_activity_score + sufficient_sleep + increased_motivation

    def get_ability(self):
        """"
        1)Chan et al (2020) "Prompto: Investigating Receptivity to Prompts Based on Cognitive Load from Memory Training
         Conversational Agent"
         "users were more receptive to prompts and memory training under low cognitive load than under high cognitive load"
        - cognitive load, high (-), low(+)


        2)self-efficacy/ confidence = positive responses rate  person who would fail in the past might be less confident
        "Bandura (1997, p. 2) has defined perceived self-efficacy as ‘the belief in one’s capabilities
        to organize and execute the courses of action required to produce given attainments.’
        Numerous studies have investigated domain-specific self-efficacy that predicts corresponding intentions,
         health behaviours, and health behaviour change (Burkert, Knoll,
        Scholz, Roigas, & Gralla, 2012; Luszczynska & Schwarzer, 2005).
        Bandura, A. (1997). Self-efficacy: The exercise of control. New York: Freeman.
        Luszczynska, A., & Schwarzer, R. (2005). Social cognitive theory. In M. Connor & P. Norman (Eds.),
        Predicting health behaviour (pp. 127–169). London: Open University Press
        "

        Other:
        task_difficulty,
        length
        sequence mining SPADE
        """

        n = self.activity_p  # 0  if the activity was already performed twice
        if n == 0:
            not_tired_of_repeating_the_activity = 1
        elif n == 1:
            not_tired_of_repeating_the_activity = 0
        else:
            not_tired_of_repeating_the_activity = -1
        ready = self._time_since_last_activity()
        load = 1 if self.cognitive_load == 0 else 0
        confidence = 1 if sum(self.activity_performed) >= self.confidence_threshold else 0

        return confidence + load + not_tired_of_repeating_the_activity + ready

    def get_trigger(self):
        """"
        1)Bidargadi et al. (2018) "To Prompt or Not to Prompt? A Microrandomized Trial of Time-Varying Push Notifications to
         Increase Proximal Engagement With a Mobile Health App"
        Timing!: "users are more likely to engage with the app within 24 hours when push notifications are sent at mid-day
         on weekends"
         - time of the day
         - day of the week

        2) Akker etal (2015) "Tailored motivational message generation: A model and practical framework for real-time
         physical activity coaching"

                                     #Trigger
        3) Goyal et al. (2017) users are likely to pay attention to the notifications at times of increasing arousal
        - arousal, high (+), low (-) Notice a Trigger? --> in a future shall model as a continuum and only in mid
         arousal effective

        4) Aminikhanghahi (2017) "Improving Smartphone Prompt Timing Through Activity Awareness"
            "participants did not like to respond to AL queries when they were at work but were generally responsive
             when they were relaxing " relaxing low cognitive load and positive valence

             Ho et al. (2018). Location = [Home, Work, Other] , Motion activity = [Stationary, Walking, Running, Driving]
         - home (+), other (-)
         - stationary(+), walking(-), driving (-)
         - awake (+) sleeping (-)

        """

        prompt = 1 if self.awake_list[-1] != 'sleeping' else 0  # do not prompt when patient sleep
        good_time = 1 if self._get_time_day() == self.good_time else 0
        good_day = 1 if self._get_week_day() == 1 else 0
        good_location = 1 if self.location == 'home' else 0
        good_motion = 1 if self.motion_activity_list[-1] == 'stationary' else 0
        good_arousal = 1 if self.arousal == 1 else 0

        return (good_arousal + good_day + good_time + good_location + good_motion) * prompt

    def update_state(self):
        self._update_time()
        self._update_awake()
        if self.awake_list[-1] == 'awake':
            self._update_motion_activity()
            self._update_location()
            self._update_emotional_state()
        else:
            self.location = 'home'
            self.motion_activity_list.append('stationary')
            self.arousal = 0
            self.cognitive_load = 0
            self.valence_list.append(self.valence)
            self.arousal_list.append(self.arousal)

    def _update_day(self):

        self.week_days.rotate(-1)
        self.day_of_the_week = self.week_days[0]

    def _get_week_day(self):
        if self.day_of_the_week < 6:
            return 0  # week day
        else:
            return 1  # weekend

    def _get_time_day(self):
        if 10 >= self.time_of_the_day >= 6:
            return 0  # morning
        elif 18 > self.time_of_the_day >= 10:
            return 1  # midday
        elif 22 > self.time_of_the_day >= 18:
            return 2  # evening
        else:
            return 3  # night

    def _update_time(self):

        self.hours.rotate(-1)
        self.time_of_the_day = self.hours[0]
        if self.time_of_the_day == 0:
            self._update_day()

    def _start_time_randomiser(self):
        for i in range(np.random.randint(0, len(self.week_days))):
            self.week_days.rotate(-1)
        for i in range(np.random.randint(0, len(self.hours))):
            self.hours.rotate(-1)

    def _update_emotional_state(self):
        # random
        self._update_patient_stress_level()  # updates arousal and valence
        self._update_patient_cognitive_load()  # 0 low, 1 high

    def _update_motion_activity(self):

        if self.activity_performed[-1] == 1:
            weights = (0, 1)
        else:
            threshold = 0.3 # equivalent to 6 h daily
            threshold += self._break_record() + self._compete_with_peers()
            w_r = self.motion_activity_list.count('walking') / len(self.motion_activity_list)
            w = w_r if w_r < threshold else threshold
            st = 1-w
            weights = (st, w)
        self.motion_activity_list.append(random.choices(['stationary', 'walking'], weights=weights, k=1)[0])

    def _break_record(self):
        if self.patient_profile == 2 or (self.patient_profile == 3 and random.random() >= 0.3):
            if self.record_broken and self.record > 0:
                self.record_broken = False
                return random.uniform(0.1, 0.4)
        return 0
    
    def _compete_with_peers(self):
        if self.patient_profile == 2 or (self.patient_profile == 3 and random.random() >= 0.5):
            act_frac = self.motion_activity_list.count('walking') / len(self.motion_activity_list)
            # we simulate the peers activities as the normal distribution assuming that an average person would walk 3
            # hours a day
            normal_dist = np.random.normal(loc=3 / 24, scale=1 / 24)
            if abs(act_frac - normal_dist) <= self.compete_threshold:
                return random.uniform(0.1, 0.4)  # increase of activity duration
        return 0

    def _update_awake(self):
        """"
        Fairholme & Manber (2015) "Sleep, emotions, and emotion regulation: an overview"
        "negative valence and high arousal potentially have unique effects on sleep architecture,
        with high arousal being associated with reductions in slow-wave sleep and negative valence being associated
        with disruptions to REM sleep"
           - negative valence (-)
           - high arousal (-)

        Bisson et al (2019) "Walk to a better night of sleep: testing the relationship between physical
        activity and sleep"
        " on days that participants were more active than average,
        they reported better sleep quality and duration in both sexes. Results suggest
        that low-impact PA is positively related to sleep, more so in women than men"
             - walking (+) """

        if self.activity_p > 0:
            awake_prb = self.health_sleep[self.time_of_the_day] # healthy sleeping
        else:
            if self.arousal == 2 and self.valence == 0:
                awake_prb = self.insomnia[self.time_of_the_day]# insomnia
            else:
                awake_prb = self.semihealthy_sleep[self.time_of_the_day]# semi-healthy

        now_awake = random.choices(['sleeping', 'awake'], weights=(1 - awake_prb, awake_prb), k=1)
        self.awake_list.append(now_awake[0])

    def _update_location(self):
        if self.motion_activity_list[-1] == 'walking':
            self.location = 'other'
        else:
            self.location = random.choices(['home', 'other'], weights=(0.8, 0.2), k=1)[0]

    @staticmethod
    def _prob_awake(x):
        x = x + 14
        return -0.5 * math.sin((x + 2) / 3.5) + 0.5

    def _awake_pattern(self, x, z):
        x = x - 14
        x = abs(x)
        return np.where(x <= 6, 0.98, self._prob_awake(x) + z)

    def _initialise_awake_probailities(self):

        self.health_sleep = [self._awake_pattern(x, 0.15) for x in range(0, 24)]
        self.semihealthy_sleep = [self._awake_pattern(x, 0.35) for x in range(0, 24)]
        self.insomnia = [self._awake_pattern(x, 0.6) for x in range(0, 24)]

    def _update_patient_stress_level(self):
        """"
        Stress = high arousal and negative valence Markova et al (2019) "arousal-valence emotion space"
        in contrast to
        Flow = high/mid arousal and positive valence//

        Peifer et al (2014) "The relation of flow-experience and physiological arousal under stress — Can u shape it?"
        "Physiological arousal during flow-experience between stress and relaxation"

        1) Yoon et al (2014) "Understanding notification stress of smartphone messenger app"
         - number of notification high (stress +), low (stress -)

        2) Zhai et al (2020) "Associations among physical activity and smartphone use with perceived stress and sleep
        quality of Chinese college students"
         - insufficient exercise (stress +), exercise in past day (stress -)
        """

        insufficient_exercise = 1 if self.motion_activity_list[-24:].count('walking') < 1 else 0
        annoyed = 1 if self.activity_s > self.max_notification_tolerated else 0
        number_of_hours_slept = self.awake_list[-24:].count('sleeping')
        insufficient_sleep = 1 if number_of_hours_slept < 7 else 0
        neg_factors = insufficient_exercise + annoyed + insufficient_sleep

        if self.motion_activity_list[-1] == 'walking':
            self.valence,  self.arousal = 1, 1
        else:
            if neg_factors >= 2:
                self.valence = 0
                self.arousal = 2
            elif neg_factors == 1:
                self.valence = random.choices([0, 1], weights=(0.5, 0.5), k=1)[0]
                self.arousal = random.choices([0, 1, 2], weights=(0.3,0.3, 0.4), k=1)[0]  
            else:
                self.valence = 1
                self.arousal = random.choices([0, 1, 2], weights=(0.3, 0.4, 0.3), k=1)[0]
        self.valence_list.append(self.valence)
        self.arousal_list.append(self.arousal)

    def _update_patient_cognitive_load(self):
        """"

        Okoshi et al (2015)  "Attelia: Reducing User’s Cognitive Load due to Interruptive Notifications on Smart Phones"
        Okoshi et al (2017) "Attention and Engagement-Awareness in the Wild: A Large-Scale Study with Adaptive Notifications"
        "notifications at detected breakpoint timing resulted in 46% lower cognitive load compared to randomly-timed
         notifications"
        """
        if self.activity_s> 0:
            self.cognitive_load = 1 if self.activity_p / self.activity_s < 0.5 else 0
        else:
            self.cognitive_load = np.random.randint(0, 1) 
            
    def _update_patients_activity_score(self):
        """"
        Williams et al (2012) "Does Affective Valence During and Immediately Following a 10-Min Walk Predict Concurrent
         and Future Physical Activity?"
         "During-behavior affect is predictive of concurrent and future physical activity behavior"

        """
        self.last_activity_score = self.valence 
    
    def _increased_motivation(self):
        if self.patient_profile == 2 or (self.patient_profile == 3 and random.random() >= 0.5):
            if self._better_than_peers():
                return 1
        return 0
    
    def _better_than_peers(self):
        act_frac = self.motion_activity_list.count('walking') / len(self.motion_activity_list)
        normal_dist = np.random.normal(loc=3 / 24, scale=1 / 24)
        if normal_dist < act_frac:
            return 1
        return 0
        
def update_patient_arousal():
    """"
    Kusserow et al (2013) "Modeling arousal phases in daily living using wearable sensors"
    "participant-specific arousal was frequently estimated during conversations and
    yielded highest intensities during office work"


    """
    pass


def update_patient_valence():
    """"
    Baglioni  et al (2010) "Sleep and emotions: A focus on insomnia"
    "interaction between sleep and emotional valence, poor sleep quality seems to
    correlate with high negative and low positive emotions, both in clinical and subclinical samples"

    - sufficient sleep (+), insufficient seep(-)

    Niedermeier et al (2021) "Acute Effects of a Single Bout of Walking on Affective
    Responses in Patients with Major Depressive Disorder"
    "positively valenced immediate response of light- to moderate-intensity walking may serve as an acute
    emotion regulation"

    Ivarsson et al (2021) "Associations between physical activity and core affects within and across days:
     a daily diary study"
     " physical activity seem to have a positive within-day association with pleasant
    core affects and a negative within-day association with unpleasant-deactivated core
    affects. There were, however, no statistically significant relations between core affects
    and physical activity across days"

    - recently walking (+),

    """
    pass




