import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";

// Define the multiply tool (from the documentation)
const multiply = tool(
  ({ a, b }) => {
    /**
     * Multiply two numbers.
     */
    return a * b;
  },
  {
    name: "multiply",
    description: "Multiply two numbers",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  }
);

// You can add more tools here
const divide = tool(
  ({ a, b }) => {
    /**
     * Divide two numbers.
     */
    if (b === 0) {
      throw new Error("Cannot divide by zero");
    }
    return a / b;
  },
  {
    name: "divide",
    description: "Divide two numbers",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  }
);

async function runToolExample() {
  // Initialize the LLM
  const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0,
    // Make sure to set your OpenAI API key in environment variables
    // or pass it directly: openAIApiKey: "your-api-key"
  });

  // Bind the tools to the LLM
  const llmWithTools = llm.bindTools([multiply, divide]);

  // Example 1: Simple tool usage
  console.log("=== Example 1: Basic Tool Usage ===");
  const response1 = await llmWithTools.invoke([
    new HumanMessage("What is 15 multiplied by 23?")
  ]);
  
  console.log("AI Response:", response1.content);
  console.log("Tool calls:", response1.tool_calls);

  // Example 2: Handle tool calls manually
  console.log("\n=== Example 2: Manual Tool Execution ===");
  const response2 = await llmWithTools.invoke([
    new HumanMessage("Calculate 144 divided by 12, then multiply the result by 7")
  ]);

  if (response2.tool_calls && response2.tool_calls.length > 0) {
    console.log("AI wants to use tools:", response2.tool_calls);
    
    // Execute the tool calls
    const toolResults = [];
    for (const toolCall of response2.tool_calls) {
      let result;
      if (toolCall.name === "multiply") {
        result = await multiply.invoke(toolCall.args);
      } else if (toolCall.name === "divide") {
        result = await divide.invoke(toolCall.args);
      }
      
      toolResults.push({
        tool: toolCall.name,
        args: toolCall.args,
        result: result
      });
      
      console.log(`${toolCall.name}(${JSON.stringify(toolCall.args)}) = ${result}`);
    }
  }

  // Example 3: Using RunnableWithMessageHistory for conversation with tools
  console.log("\n=== Example 3: Conversational Tool Usage ===");
  
  const messages = [
    new HumanMessage("I need to calculate the area of a rectangle that is 25 units wide and 18 units tall")
  ];

  const response3 = await llmWithTools.invoke(messages);
  console.log("AI Response:", response3.content);
  
  if (response3.tool_calls && response3.tool_calls.length > 0) {
    for (const toolCall of response3.tool_calls) {
      if (toolCall.name === "multiply") {
        const result = await multiply.invoke(toolCall.args);
        console.log(`Area calculation: ${toolCall.args.a} × ${toolCall.args.b} = ${result} square units`);
      }
    }
  }
}

// Example 4: Error handling
async function runWithErrorHandling() {
  console.log("\n=== Example 4: Error Handling ===");
  
  const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0,
  });

  const llmWithTools = llm.bindTools([multiply, divide]);

  try {
    const response = await llmWithTools.invoke([
      new HumanMessage("What is 10 divided by 0?")
    ]);
    
    if (response.tool_calls && response.tool_calls.length > 0) {
      for (const toolCall of response.tool_calls) {
        try {
          if (toolCall.name === "divide") {
            const result = await divide.invoke(toolCall.args);
            console.log(`Result: ${result}`);
          }
        } catch (error) {
          console.error(`Tool execution error: ${error.message}`);
        }
      }
    }
  } catch (error) {
    console.error(`LLM error: ${error.message}`);
  }
}

// Run the examples
async function main() {
  try {
    await runToolExample();
    await runWithErrorHandling();
  } catch (error) {
    console.error("Error running examples:", error);
    console.log("\nMake sure to:");
    console.log("1. Install dependencies: npm install @langchain/openai @langchain/core zod");
    console.log("2. Set your OpenAI API key: export OPENAI_API_KEY=your-key-here");
  }
}

// Uncomment to run
main();