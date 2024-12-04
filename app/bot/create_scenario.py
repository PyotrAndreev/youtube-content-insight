from g4f.client import Client

def generate_scenario(topic):
    print("running...")
    print(topic)
    client = Client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Напиши сценарий видео на тему " + topic}],
        # Add any other necessary parameters
    )
    print(1)
    scenario = (response.choices[0].message.content)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Добавь неожиданный сюжетный поворот в сценарий  " + scenario}],
        # Add any other necessary parameters
    )

    # new_scenario = (response.choices[0].message.content)

    scenario = (response.choices[0].message.content)
    print(scenario)
    return scenario