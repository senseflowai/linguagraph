import subprocess
import json
import sys
import os

def main():
    # 1. Получаем текстовый вопрос от пользователя
    user_query = "Сколько камер было создано вчера"

    # 2. Генерируем промпт для OpenAI
    # Используем команду linguagraph prompt для получения системного промпта
    try:
        prompt_result = subprocess.run(
            ["./target/debug/linguagraph", "prompt"],
            capture_output=True,
            text=True,
            check=True
        )
        system_prompt = prompt_result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при получении промпта: {e.stderr}")
        sys.exit(1)

    # 3. Формируем полный запрос для OpenAI API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    # 4. Отправляем запрос в OpenAI API (используем curl как в примере)
    # ВАЖНО: Замените YOUR_API_KEY на ваш реальный API ключ OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")  # Или захардкодьте, но не рекомендуется

    curl_command = [
        "curl", "-s", "-X", "POST", "http://100.79.136.128:8080/v1/chat/completions",
        "-H", f"Content-Type: application/json",
        # "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps({
            "model": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "messages": messages,
            "temperature": 0.1
        })
    ]

    try:
        response_result = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            check=True
        )
        response_data = json.loads(response_result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запросе к OpenAI API: {e.stderr}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON от OpenAI: {response_result.stdout}")
        sys.exit(1)


    # 6. Извлекаем текст ответа из JSON
    try:
        assistant_message = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"Ошибка при извлечении ответа из JSON: {e}")
        print(f"Полный ответ: {response_data}")
        sys.exit(1)

    # 7. Создаем JSON файл для linguagraph run
    # Предположим, что linguagraph run ожидает JSON с полем "query" или подобным
    # Вам нужно будет адаптировать эту часть под формат, который ожидает linguagraph run
    linguagraph_input = json.loads(assistant_message)

    linguagraph_file = "linguagraph_input.json"
    with open(linguagraph_file, 'w', encoding='utf-8') as f:
        json.dump(linguagraph_input, f, ensure_ascii=False, indent=2)

    print(f"Файл для linguagraph run создан: {linguagraph_file}")

    # 8. Вызываем linguagraph run с созданным JSON файлом
    try:
        run_result = subprocess.run(
            ["./target/debug/linguagraph", "run", linguagraph_file],
            capture_output=True,
            text=True,
            check=True
        )
        print("\n--- Ответ от linguagraph run ---")
        print(run_result.stdout)
        if run_result.stderr:
            print("\n--- Ошибки/предупреждения ---")
            print(run_result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении linguagraph run: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()