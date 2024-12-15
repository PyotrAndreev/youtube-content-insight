from g4f.client import Client


def generate_scenario(topic):
    print("running...")
    print(topic)
    client = Client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Представь, что ты - профессиональный режиссер, Напиши сценарий видео на,который должен включать в себя завязку, кульминацию и развязку. Уложись в 2000 символов. Тема видео:" + topic}],
        # Add any other necessary parameters
    )
    print(1)
    scenario = (response.choices[0].message.content)

    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": "Добавь неожиданный сюжетный поворот в сценарий  " + scenario}],
    #     # Add any other necessary parameters
    # )
    #
    # # new_scenario = (response.choices[0].message.content)
    #
    # scenario = (response.choices[0].message.content)
    print(scenario)
    return scenario