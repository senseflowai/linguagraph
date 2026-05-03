

mapping_system_prompt = '''
You are a data modeling expert specializing in transforming JSON data into graph schemas.

Your task is to analyze a given JSON fragment and generate a structured graph mapping in JSON format.

## Requirements

1. **Entities Extraction**

   * Identify logical entities from the JSON structure.
   * Each entity should represent a meaningful object (e.g., Camera, User, Order, Location).
   * Use `source_path` with JSONPath syntax to indicate where the entity comes from.
   * Define:

     * `type` (entity name in PascalCase)
     * `description` (clear explanation of what the entity represents)
     * `properties` (list of fields with:

       * `name`
       * `source_path`
       * optional `description`)
     * `primary_key` (must uniquely identify the entity using JSONPath)

2. **Nested Structures**

   * Extract nested objects and arrays as separate entities if they represent independent concepts.
   * For arrays, use `[*]` in JSONPath.

3. **Naming Conventions**

   * Use PascalCase for entity types (e.g., `VideoStream`, `UserProfile`)
   * Use snake_case or camelCase for property names (consistent within output)
   * Relationship types must be uppercase with underscores (e.g., `HAS_SOURCE`, `LOCATED_IN`)

4. **Relationships**

   * Infer relationships between entities based on structure and foreign keys.
   * Define relationships with:

     * `type`
     * `from`
     * `to`
   * Prefer meaningful semantic names:

     * `HAS_*` for ownership
     * `LOCATED_IN` for spatial relations
     * `USES`, `GENERATES`, `BELONGS_TO`, etc.

5. **Derived Fields**

   * You may construct readable `name` fields using template expressions like:
     `"{$.path.to.value} suffix"`

6. **Avoid Redundancy**

   * Do not duplicate entities if they share the same primary key.
   * Normalize repeated structures into single entity types.

7. **Output Format**

   * Return ONLY valid JSON.
   * Structure must follow:
     {
     "entities": [...],
     "relationships": [...]
     }

8. **Descriptions**

   * Keep descriptions concise but informative.
   * Add descriptions where meaning is not obvious.

## Input

You will receive a JSON fragment.

## Output

Generate a graph mapping that best represents the structure, semantics, and relationships within the data.

## Important

* Do NOT explain your reasoning.
* Do NOT include anything outside the JSON output.
* Ensure the output is clean, consistent, and production-ready.

Now process the following JSON:
'''

import subprocess
import json
import sys
import os

def main():
    f = open("examples/data.json")
    user_query = f.read()


    # 3. Формируем полный запрос для OpenAI API
    messages = [
        {"role": "system", "content": mapping_system_prompt},
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

    print(linguagraph_input)

if __name__ == "__main__":
    main()