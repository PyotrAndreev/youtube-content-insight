# youtube-content-insight
AI-инструмент для анализа и управления комментариями к видеоконтенту на YouTube.

## Запуск проекта(в формате тг-бота)

- `git clone`
- `mv .env.example .env` + подставить свои значения в env
- `cd ./youtube-content-insight`
- `docker build -t bot .` - билд контейнера бота
- `docker compose up -d` - paзвертка приложения

## О проекте

Наш проект предоставляет возможность:

- Анализировать комментарии: Кластеризация комментариев позволяет выделить самые популярные тематики и отправить пользователю самые яркие комментарии из каждого кластера. Это помогает понять, что именно интересует зрителей, без необходимости читать большое количество комментариев самостоятельно.

- Отслеживать динамику комментариев: Пользователь может наблюдать, как изменяется количество положительных и отрицательных комментариев в нескольких видео. Это дает возможность понять, какие видео больше нравятся аудитории, а какие вызывают негативную реакцию. Такой анализ может быть полезен для блогеров, желающих экспериментировать с форматами и тематиками видео.

- Изучать популярные теги: Теги являются одним из самых широко используемых инструментов для продвижения на YouTube. Наш инструмент позволяет узнать, какие теги чаще всего используются в самых популярных видео по заданной тематике (тематика определяется автоматически YouTube).

- Получение списка популярных видео: Пользователь может получить список 10 самых популярных видео из заданной тематики.

- Генерация сценариев: Наш инструмент взаимодействует с ChatGPT, позволяя генерировать сценарии видео по заданной тематике, используя определенную схему для создания сценариев, которые будут действительно интересны зрителям.

---

## Преимущества и отличия от аналогов

### 1. **Комплексный подход**
Большинство инструментов на рынке фокусируются на одной функции, например, на анализе тональности или сборе статистики. Наш проект объединяет полный спектр возможностей: от аналитики комментариев до генерации сценариев и тегов для продвижения.

### 2. **Новизна в интеграции анализа и генерации контента**
Мы не просто анализируем данные, но и превращаем их в actionable insights. Например:
- Анализ комментариев выявляет, какие темы интересуют аудиторию.
- С помощью ИИ создается черновик сценария нового видео, основанный на популярных запросах и обсуждениях.

### 3. **Глубокий анализ аудитории**
Наш инструмент предоставляет более детализированную аналитику:
- Выявляет скрытые тренды и паттерны в комментариях благодаря кластеризации.
- Использует контекстуальный анализ для определения тональности, учитывая сарказм и неоднозначные формулировки.

### 4. **Автоматизация продвижения**
Мы автоматически генерируем список популярных и релевантных тегов, что позволяет создателям контента привлекать больше зрителей и повышать видимость видео.

### 5. **Удобство и доступность**
Подробные отчеты в удобном формате, которые можно использовать для презентаций или внутреннего анализа.

### 6. **Поддержка русскоязычной аудитории**
Наш инструмент разработан с учетом особенностей русского языка, включая сложные грамматические структуры и идиоматические выражения.

### Итог:

- **Гибридная модель анализа и генерации контента**: это не просто аналитика, а готовые рекомендации для создания следующего видеоролика.
- **Использование кластеризации тем**: помогает авторам глубже понять интересы аудитории.
- **Учет специфики языка**: проект идеально подходит для работы с русскоязычными комментариями.
- **Автоматическое предложение тегов**: экономит время на подбор ключевых слов для продвижения.

---

## Стек технологий:

Python, FastApi, HuggingFace, aiogram, requests, logging, sqlalchemy, sklearn, pandas, numpy, matplotlib

## Этапы:
1. **Группировка и Сводка**
2. **Интеграция и Оптимизация**
3. **Фильтрация**
4. **Динамический Анализ**
5. **Анализ Аудитории**
6. **Статистика и Опросы**
