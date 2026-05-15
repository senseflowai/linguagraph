import subprocess
import json
import sys
import os
import locale


def decode_stdin_line(raw_line):
    encodings = [
        sys.stdin.encoding,
        locale.getpreferredencoding(False),
        "utf-8",
        "cp1251",
    ]

    for encoding in dict.fromkeys(filter(None, encodings)):
        try:
            return raw_line.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue

    return raw_line.decode("utf-8", errors="replace")


def read_user_query(prompt):
    if not hasattr(sys.stdin, "buffer"):
        return input(prompt).strip()

    print(prompt, end="", flush=True)
    raw_line = sys.stdin.buffer.readline()
    if raw_line == b"":
        raise EOFError

    return decode_stdin_line(raw_line).strip()


def get_system_prompt():
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

    return system_prompt


def process_query(user_query, system_prompt):
    # 1. Формируем полный запрос для OpenAI API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    # 2. Отправляем запрос в OpenAI API (используем curl как в примере)
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
        return


    # 3. Извлекаем текст ответа из JSON
    try:
        assistant_message = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"Ошибка при извлечении ответа из JSON: {e}")
        print(f"Полный ответ: {response_data}")
        return

    # 4. Создаем JSON файл для linguagraph run
    # Предположим, что linguagraph run ожидает JSON с полем "query" или подобным
    # Вам нужно будет адаптировать эту часть под формат, который ожидает linguagraph run
    assistant_message = assistant_message.replace('```json', '').replace('```', '').strip()
    try:
        linguagraph_input = json.loads(assistant_message)
    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON от ассистента: {assistant_message}")
        return

    linguagraph_file = "linguagraph_input.json"
    with open(linguagraph_file, 'w', encoding='utf-8') as f:
        json.dump(linguagraph_input, f, ensure_ascii=False, indent=2)

    # 5. Вызываем linguagraph run с созданным JSON файлом
    try:
        run_result = subprocess.run(
            ["./target/debug/linguagraph", "run", linguagraph_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(run_result.stdout)
        if run_result.stderr:
            print("\n--- Ошибки/предупреждения ---")
            print(run_result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении linguagraph run: {e.stderr}")


def main():
    # Генерируем системный промпт один раз и переиспользуем для каждого запроса.
    system_prompt = get_system_prompt()

    print("Введите user_query. Ctrl-D или Ctrl-C для выхода.")
    while True:
        try:
            user_query = read_user_query("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_query:
            continue

        process_query(user_query, system_prompt)

if __name__ == "__main__":
    main()
