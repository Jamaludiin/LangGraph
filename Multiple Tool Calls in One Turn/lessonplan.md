🧠 3️⃣ Multiple Tool Calls in One Turn

Right now you handle:

response.tool_calls[0]


But LLM can return:

response.tool_calls = [
  {...divide...},
  {...subtract...}
]


You need to loop through all tool calls.

This is important for:

"What is 45 divided by 9 and 20 minus 5?"

Real agents must handle multiple tool calls.