import time
from datetime import datetime
import pandas as pd
from google import genai
from google.genai import types

import matplotlib.pyplot as plt

client = genai.Client(api_key="AIzaSyCTOx8Cb_hyi1LHYK5fEmDpB6RQzqhULI8")
system_instruction = """You are a performance evaluation analyzer.
You will receive: The name of a physical exercise, The name of the joint angle being monitored (e.g., shoulder, elbow),
A list of joint angle values by frames recorded across 8 reptitions of the exercise of the robotic trainer. 
A list of joint angle values by frames recorded across 8 repetitions of the exercise of the trainee.
Your task is to determine whether the trainee performed the exercise correctly and/or consistently.
Take into account that the trainee motion is a human motion montiored by a camera which may have deviations, and the trainer motion is the absolute value of the robot motions.
Ignore frames at the very beginning or end of the sequence if they appear to capture idle time before/after the motion actually occurred.
Return a clear evluation and explain briefly.
following scheme:
Evaluation: Correct/Incorrect ; Consistence/Inconsistence
Explanation: "".
 """

data = pd.read_csv(r"CSV/Raw Data/ODS_YDS_all_raw_data.csv")
demo_values = [0.5999999999999943, 38.8, 89.4, 86.5, 86.8, 86.9, 87.0, 87.1, 87.2, 87.2, 87.3, 87.4, 87.4, 87.5, 20.200000000000003, 3.299999999999997, 2.0999999999999943, 2.0999999999999943, 2.0, 2.0, 1.9000000000000057, 1.9000000000000057, 1.7999999999999972, 1.7999999999999972, 1.7999999999999972, 1.7000000000000028, 1.7000000000000028, 1.5999999999999943, 1.5999999999999943, 23.599999999999994, 89.7, 86.9, 87.8, 87.9, 87.9, 88.0, 88.0, 88.1, 88.1, 88.2, 88.2, 88.3, 20.799999999999997, 3.9000000000000057, 2.5999999999999943, 2.5999999999999943, 2.5, 2.5, 2.4000000000000057, 2.4000000000000057, 2.299999999999997, 2.299999999999997, 2.200000000000003, 2.200000000000003, 2.0999999999999943, 2.0, 2.0, 16.400000000000006, 92.0, 87.3, 88.2, 88.3, 88.3, 88.4, 88.4, 88.4, 88.5, 88.5, 88.6, 88.6, 45.5, 0.4000000000000057, 2.700000000000003, 2.9000000000000057, 2.799999999999997, 2.700000000000003, 2.700000000000003, 2.5999999999999943, 2.5, 2.5, 2.4000000000000057, 2.4000000000000057, 2.299999999999997, 2.299999999999997, 2.299999999999997, 32.1, 91.8, 87.5, 88.4, 88.5, 88.5, 88.5, 88.6, 88.6, 88.6, 88.7, 88.7, 88.7, 45.6, 4.200000000000003, 2.9000000000000057, 2.9000000000000057, 2.9000000000000057, 2.799999999999997, 2.700000000000003, 2.700000000000003, 2.5999999999999943, 2.5, 2.5, 2.4000000000000057, 2.4000000000000057, 2.299999999999997, 2.299999999999997, 24.200000000000003, 91.8, 88.2, 88.4, 88.5, 88.6, 88.6, 88.6, 88.7, 88.7, 88.7, 88.8, 88.8, 21.200000000000003, 4.299999999999997, 3.0999999999999943, 3.0, 2.9000000000000057, 2.9000000000000057, 2.799999999999997, 2.700000000000003, 2.700000000000003, 2.5999999999999943, 2.5, 2.5, 2.4000000000000057, 2.299999999999997, 2.299999999999997, 16.700000000000003, 90.4, 87.5, 88.6, 88.5, 88.6, 88.6, 88.6, 88.6, 88.7, 88.7, 88.7, 88.7, 57.6, 0.5, 3.200000000000003, 3.0, 2.9000000000000057, 2.799999999999997, 2.799999999999997, 2.700000000000003, 2.5999999999999943, 2.5999999999999943, 2.5, 2.5, 2.4000000000000057, 2.299999999999997, 2.299999999999997, 32.2, 87.6, 87.5, 88.7, 88.5, 88.6, 88.6, 88.6, 88.7, 88.7, 88.7, 88.8, 88.8, 57.6, -2.5, 4.200000000000003, 2.9000000000000057, 3.0, 2.9000000000000057, 2.9000000000000057, 2.799999999999997, 2.700000000000003, 2.700000000000003, 2.5999999999999943, 2.5999999999999943, 2.5, 2.4000000000000057]
demo_values = [0.29999999999999716, 0.29999999999999716, 8.099999999999994, 30.5, 49.5, 64.7, 79.6, 89.6, 89.4, 87.2, 86.1, 85.4, 86.0, 86.6, 81.6, 59.9, 28.6, -2.200000000000003, -4.700000000000003, 0.0, 2.200000000000003, 1.7999999999999972, 1.0, 0.7000000000000028, 0.7999999999999972, 1.0, 1.0, 1.0, 1.0, 1.7000000000000028, 21.5, 40.7, 59.1, 76.4, 88.2, 90.9, 89.3, 86.6, 86.2, 86.3, 87.1, 87.3, 87.2, 73.0, 46.3, 10.700000000000003, -5.599999999999994, -3.5, 2.299999999999997, 2.799999999999997, 2.0, 1.4000000000000057, 1.4000000000000057, 1.5, 1.5999999999999943, 1.5999999999999943, 1.5999999999999943, 2.799999999999997, 17.400000000000006, 33.1, 58.0, 72.7, 87.6, 91.3, 90.5, 88.0, 86.8, 86.9, 87.7, 87.9, 85.0, 65.5, 34.7, 7.400000000000006, -4.5, -3.0999999999999943, 1.5999999999999943, 3.4000000000000057, 2.700000000000003, 1.7999999999999972, 1.9000000000000057, 2.0, 2.0999999999999943, 2.0, 2.0, 2.700000000000003, 16.400000000000006, 35.1, 56.8, 74.4, 88.6, 92.0, 90.3, 88.4, 87.3, 87.4, 88.0, 88.4, 80.6, 61.5, 25.299999999999997, 2.9000000000000057, -4.700000000000003, 1.0, 3.5, 3.4000000000000057, 2.799999999999997, 2.299999999999997, 2.299999999999997, 2.4000000000000057, 2.4000000000000057, 2.4000000000000057, 2.4000000000000057, 3.0999999999999943, 16.700000000000003, 27.4, 49.5, 73.3, 88.3, 92.4, 91.1, 88.9, 87.8, 87.8, 88.3, 88.8, 81.0, 61.9, 20.799999999999997, -2.799999999999997, -4.200000000000003, 1.7000000000000028, 4.0, 3.5999999999999943, 2.700000000000003, 2.5, 2.700000000000003, 2.799999999999997, 2.799999999999997, 2.700000000000003, 2.700000000000003, 4.599999999999994, 18.5, 42.0, 60.4, 78.9, 90.7, 92.7, 90.1, 88.5, 87.9, 88.6, 88.9, 89.1, 81.3, 59.9, 28.4, -1.0999999999999943, -3.799999999999997, 1.5999999999999943, 4.400000000000006, 4.299999999999997, 3.299999999999997, 2.799999999999997, 2.799999999999997, 3.0, 3.0, 3.0, 3.0, 4.200000000000003, 13.099999999999994, 29.5, 49.9, 76.5, 88.1, 92.9, 91.3, 90.2, 88.2, 88.6, 89.1, 89.4, 88.0, 66.9, 36.0, 6.900000000000006, -3.799999999999997, -0.5999999999999943, 3.799999999999997, 4.5, 3.5999999999999943, 3.0, 3.0999999999999943, 3.200000000000003, 3.200000000000003, 3.200000000000003]

demo_values = [f - min(demo_values) for f in demo_values]
plt.figure()
x = list(range(len(demo_values)))
plt.scatter(x, demo_values)
plt.show()

for i in range(80, 90):
  print(f"Participant: {data.iloc[i][0]}, Source: {data.iloc[i][1]}, Exercise: {data.iloc[i][2]}, Hand:{data.iloc[i][3]}")
  sample = data.iloc[i][4:].dropna()

  x = list(range(len(sample)))
  plt.scatter(x, sample)
  plt.show()

  sample = sample - min(sample)
  sample_values = ", ".join(str(x) for x in sample)

  content = f"exercise: {data.iloc[i][2]}, angle name: armpit, trainer anggle values: {demo_values}," \
            f" trainee angle values: {sample_values}"

  response = client.models.generate_content(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(
          system_instruction=system_instruction),
      contents=content
  )
  print(response.text)
  time.sleep(1)



# chat like

data = pd.read_csv(r"C:\Users\mayak\PycharmProjects\DataAnalysis\CSV\Raw Data\maya_raise_arms_horizontally.csv")
sample = data.iloc[0][2:].dropna()
sample_values = ", ".join(str(x) for x in sample)

client = genai.Client(api_key="AIzaSyCTOx8Cb_hyi1LHYK5fEmDpB6RQzqhULI8")
chat = client.chats.create(model="gemini-1.5-flash", config=types.GenerateContentConfig(system_instruction=system_instruction))

content = f"exercise: raise arm horiznotally, angle name: armpit, angle values: {sample_values}"

response = chat.send_message(content)
print(response.text)




"""
An exercise is considered correct and consistent if:
Consistency: All 8 repetitions follow a similar movement pattern, with similar start, peak, and end angles.
The movement is dynamic (e.g., raising/lowering an arm, moving side to side) — not static."""


import tiktoken
