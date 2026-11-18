- Add Claude as a typed-chat AI Provider. It answers in chat only through
  `/api/anthropic/chat` with `ANTHROPIC_API_KEY` and `ANTHROPIC_CHAT_MODEL`
  (default `claude-opus-5-5`, effort `ANTHROPIC_CHAT_EFFORT`, default `low`); with Claude selected, the mic button opens chat
  instead of starting voice (thenekas, #1590).
