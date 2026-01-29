#!/usr/bin/env python3
"""

cd "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT" && source .venv/bin/activate && python test_agents.py <<< "1
b"

"""

"""
🧪 Individual Agent Tester
Test each agent node one by one to see their output.
Run: python test_agents.py
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Colors for terminal output
# ============================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


def print_output(title: str, content: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}📤 {title}:{Colors.END}")
    print(f"{'-'*50}")
    # Limit output length for readability
    if len(content) > 3000:
        print(content[:3000])
        print(f"\n{Colors.YELLOW}... (truncated, {len(content)} total chars){Colors.END}")
    else:
        print(content)
    print(f"{'-'*50}\n")


# ============================================================
# State Management
# ============================================================
class AgentState:
    """Manages the state across agent tests"""
    
    def __init__(self):
        self.data = {
            "code_under_test": "",
            "test_plan": "",
            "test_case": "",
            "static_analyzer": "",
            "patch_suggestion": "",
            "execution": "",
            "documentation_generator": "",
            "coverage_enhancer": "",
            "mutational_testing": "",
            "bug_predictor": "",
            "test_plan_reviewer": "",
            "test_case_optimizer": ""
        }
        self.output_dir = os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_result(self, agent_name: str, result: str):
        """Save agent result to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_name}_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(f"# {agent_name} Output\n\n")
            f.write(f"**Time:** {datetime.now().isoformat()}\n\n")
            f.write(f"## Result\n\n```\n{result}\n```")
        
        print_info(f"Saved to: {filepath}")
    
    def show_status(self):
        """Show current state"""
        print_header("Current State Status")
        for key, value in self.data.items():
            status = f"{Colors.GREEN}✅ Set ({len(value)} chars){Colors.END}" if value else f"{Colors.YELLOW}⬜ Empty{Colors.END}"
            print(f"  {key}: {status}")


# ============================================================
# Agent Test Functions
# ============================================================

def test_code_reader(state: AgentState):
    """Test the Code Reader Agent"""
    print_header("1️⃣ Code Reader Agent")
    
    from agents.code_reader_node import code_reader_node
    
    print_info("Reading code from programUnderTest/put.py...")
    
    result = code_reader_node(state.data)
    state.data["code_under_test"] = result["code_under_test"]
    
    print_success("Code read successfully!")
    print_output("Code Under Test", result["code_under_test"])
    state.save_result("code_reader", result["code_under_test"])
    
    return True


def test_static_analyzer(state: AgentState):
    """Test the Static Analyzer Agent"""
    print_header("2️⃣ Static Analyzer Agent")
    
    if not state.data["code_under_test"]:
        print_error("Need code_under_test first! Run Code Reader (1) first.")
        return False
    
    from agents.static_analyzer_node import static_analyzer_node
    
    print_info("Analyzing code structure...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = static_analyzer_node(state.data)
    state.data["static_analyzer"] = result["static_analyzer"]
    
    print_success("Static analysis complete!")
    print_output("Analysis Result", result["static_analyzer"])
    state.save_result("static_analyzer", result["static_analyzer"])
    
    return True


def test_test_plan(state: AgentState):
    """Test the Test Plan Generator Agent"""
    print_header("3️⃣ Test Plan Generator Agent")
    
    if not state.data["code_under_test"]:
        print_error("Need code_under_test first! Run Code Reader (1) first.")
        return False
    
    from agents.test_plan_node import test_plan_node
    
    print_info("Generating test plan...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = test_plan_node(state.data)
    state.data["test_plan"] = result["test_plan"]
    
    print_success("Test plan generated!")
    print_output("Test Plan", result["test_plan"])
    state.save_result("test_plan", result["test_plan"])
    
    return True


def test_test_plan_reviewer(state: AgentState):
    """Test the Test Plan Reviewer Agent"""
    print_header("4️⃣ Test Plan Reviewer Agent")
    
    if not state.data["test_plan"]:
        print_error("Need test_plan first! Run Test Plan Generator (3) first.")
        return False
    
    from agents.test_plan_reviewer_node import test_plan_reviewer_node
    
    print_info("Reviewing test plan...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = test_plan_reviewer_node(state.data)
    state.data["test_plan_reviewer"] = result["test_plan_reviewer"]
    
    print_success("Test plan reviewed!")
    print_output("Review Result", result["test_plan_reviewer"])
    state.save_result("test_plan_reviewer", result["test_plan_reviewer"])
    
    return True


def test_test_case_generator(state: AgentState):
    """Test the Test Case Generator Agent"""
    print_header("5️⃣ Test Case Generator Agent")
    
    if not state.data["code_under_test"] or not state.data["test_plan"]:
        print_error("Need code_under_test and test_plan first!")
        print_info("Run: Code Reader (1) → Test Plan (3)")
        return False
    
    from agents.test_case_generator_node import test_case_generator_node
    
    print_info("Generating test cases...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = test_case_generator_node(state.data)
    state.data["test_case"] = result["test_case"]
    
    print_success("Test cases generated!")
    print_output("Generated Test Code", result["test_case"])
    state.save_result("test_case_generator", result["test_case"])
    
    return True


def test_test_case_optimizer(state: AgentState):
    """Test the Test Case Optimizer Agent"""
    print_header("6️⃣ Test Case Optimizer Agent")
    
    if not state.data["test_case"]:
        print_error("Need test_case first! Run Test Case Generator (5) first.")
        return False
    
    from agents.test_case_optimizer_node import test_case_optimizer_node
    
    print_info("Optimizing test cases...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = test_case_optimizer_node(state.data)
    state.data["test_case_optimizer"] = result["test_case_optimizer"]
    
    print_success("Test cases optimized!")
    print_output("Optimized Test Code", result["test_case_optimizer"])
    state.save_result("test_case_optimizer", result["test_case_optimizer"])
    
    return True


def test_bug_predictor(state: AgentState):
    """Test the Bug Predictor Agent"""
    print_header("7️⃣ Bug Predictor Agent")
    
    if not state.data["code_under_test"]:
        print_error("Need code_under_test first! Run Code Reader (1) first.")
        return False
    
    from agents.bug_predictor_node import bug_predictor_node
    
    print_info("Predicting potential bugs...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = bug_predictor_node(state.data)
    state.data["bug_predictor"] = result["bug_predictor"]
    
    print_success("Bug prediction complete!")
    print_output("Predicted Bugs", result["bug_predictor"])
    state.save_result("bug_predictor", result["bug_predictor"])
    
    return True


def test_execution(state: AgentState):
    """Test the Execution Agent"""
    print_header("8️⃣ Execution Agent")
    
    if not state.data["test_case"]:
        print_error("Need test_case first! Run Test Case Generator (5) first.")
        return False
    
    from agents.execution_node import execution_node
    
    print_info("Simulating test execution...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = execution_node(state.data)
    state.data["execution"] = result["execution"]
    
    print_success("Execution simulation complete!")
    print_output("Execution Results", result["execution"])
    state.save_result("execution", result["execution"])
    
    return True


def test_patch_suggestion(state: AgentState):
    """Test the Patch Suggestion Agent"""
    print_header("9️⃣ Patch Suggestion Agent")
    
    if not state.data["execution"]:
        print_error("Need execution results first! Run Execution Agent (8) first.")
        return False
    
    from agents.patch_suggestion_node import patch_suggestion_node
    
    print_info("Generating patch suggestions...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = patch_suggestion_node(state.data)
    state.data["patch_suggestion"] = result["patch_suggestion"]
    
    print_success("Patch suggestions generated!")
    print_output("Suggested Patches", result["patch_suggestion"])
    state.save_result("patch_suggestion", result["patch_suggestion"])
    
    return True


def test_mutational_testing(state: AgentState):
    """Test the Mutational Testing Agent"""
    print_header("🔟 Mutational Testing Agent")
    
    if not state.data["code_under_test"]:
        print_error("Need code_under_test first! Run Code Reader (1) first.")
        return False
    
    from agents.mutational_testing_node import mutational_testing_node
    
    print_info("Creating code mutations...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = mutational_testing_node(state.data)
    state.data["mutational_testing"] = result["mutational_testing"]
    
    print_success("Mutational testing complete!")
    print_output("Mutations & Analysis", result["mutational_testing"])
    state.save_result("mutational_testing", result["mutational_testing"])
    
    return True


def test_coverage_enhancer(state: AgentState):
    """Test the Coverage Enhancer Agent"""
    print_header("1️⃣1️⃣ Coverage Enhancer Agent")
    
    if not state.data["code_under_test"] or not state.data["test_case"]:
        print_error("Need code and test cases first!")
        return False
    
    from agents.coverage_enhancer_node import coverage_enhancer_node
    
    print_info("Analyzing coverage gaps...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = coverage_enhancer_node(state.data)
    state.data["coverage_enhancer"] = result["coverage_enhancer"]
    
    print_success("Coverage analysis complete!")
    print_output("Coverage Enhancement", result["coverage_enhancer"])
    state.save_result("coverage_enhancer", result["coverage_enhancer"])
    
    return True


def test_documentation_generator(state: AgentState):
    """Test the Documentation Generator Agent"""
    print_header("1️⃣2️⃣ Documentation Generator Agent")
    
    if not state.data["code_under_test"]:
        print_error("Need code_under_test first! Run Code Reader (1) first.")
        return False
    
    from agents.documentation_generator_node import documentation_generator_node
    
    print_info("Generating documentation...")
    print(f"{Colors.YELLOW}⏳ Calling LLM...{Colors.END}")
    
    result = documentation_generator_node(state.data)
    state.data["documentation_generator"] = result["documentation_generator"]
    
    print_success("Documentation generated!")
    print_output("Generated Documentation", result["documentation_generator"])
    state.save_result("documentation_generator", result["documentation_generator"])
    
    return True


def run_basic_pipeline(state: AgentState):
    """Run the basic pipeline: Code Reader → Test Plan → Test Cases"""
    print_header("🚀 Running Basic Pipeline")
    
    print_info("Step 1/3: Reading code...")
    test_code_reader(state)
    
    print_info("Step 2/3: Generating test plan...")
    test_test_plan(state)
    
    print_info("Step 3/3: Generating test cases...")
    test_test_case_generator(state)
    
    print_success("Basic pipeline complete!")


def run_full_pipeline(state: AgentState):
    """Run all agents in sequence"""
    print_header("🚀 Running Full Pipeline (All Agents)")
    
    agents = [
        ("Code Reader", test_code_reader),
        ("Static Analyzer", test_static_analyzer),
        ("Test Plan", test_test_plan),
        ("Test Plan Reviewer", test_test_plan_reviewer),
        ("Test Case Generator", test_test_case_generator),
        ("Test Case Optimizer", test_test_case_optimizer),
        ("Bug Predictor", test_bug_predictor),
        ("Execution", test_execution),
        ("Patch Suggestion", test_patch_suggestion),
        ("Mutational Testing", test_mutational_testing),
        ("Coverage Enhancer", test_coverage_enhancer),
        ("Documentation Generator", test_documentation_generator),
    ]
    
    for i, (name, func) in enumerate(agents, 1):
        print(f"\n{Colors.CYAN}[{i}/{len(agents)}] Running {name}...{Colors.END}")
        try:
            func(state)
        except Exception as e:
            print_error(f"Failed: {e}")
            cont = input("Continue? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    print_success("Full pipeline complete!")
    state.show_status()


# ============================================================
# Menu
# ============================================================

MENU = """
{header}╔══════════════════════════════════════════════════════════╗
║              🧪 AGENT NODE TESTER                        ║
╚══════════════════════════════════════════════════════════╝{end}

{cyan}Individual Agents:{end}
  {yellow}1{end}. Code Reader         - Read source code from put.py
  {yellow}2{end}. Static Analyzer     - Analyze code structure
  {yellow}3{end}. Test Plan           - Generate test plan
  {yellow}4{end}. Test Plan Reviewer  - Review test plan
  {yellow}5{end}. Test Case Generator - Generate pytest code
  {yellow}6{end}. Test Case Optimizer - Optimize test code
  {yellow}7{end}. Bug Predictor       - Predict potential bugs
  {yellow}8{end}. Execution           - Simulate test execution
  {yellow}9{end}. Patch Suggestion    - Suggest code fixes
  {yellow}10{end}. Mutational Testing  - Create code mutations
  {yellow}11{end}. Coverage Enhancer   - Find coverage gaps
  {yellow}12{end}. Doc Generator       - Generate documentation

{green}Pipelines:{end}
  {yellow}b{end}. Basic Pipeline (1 → 3 → 5)
  {yellow}f{end}. Full Pipeline (all agents)

{blue}Utilities:{end}
  {yellow}s{end}. Show current state
  {yellow}c{end}. Clear state
  {yellow}q{end}. Quit
""".format(
    header=Colors.HEADER + Colors.BOLD,
    end=Colors.END,
    cyan=Colors.CYAN,
    yellow=Colors.YELLOW,
    green=Colors.GREEN,
    blue=Colors.BLUE
)


def main():
    state = AgentState()
    
    print_header("Welcome to Agent Node Tester!")
    print_info(f"Results will be saved to: {state.output_dir}")
    print_info(f"Model: {os.getenv('MODEL_NAME', 'llama3-70b-8192')}")
    
    agents_map = {
        '1': test_code_reader,
        '2': test_static_analyzer,
        '3': test_test_plan,
        '4': test_test_plan_reviewer,
        '5': test_test_case_generator,
        '6': test_test_case_optimizer,
        '7': test_bug_predictor,
        '8': test_execution,
        '9': test_patch_suggestion,
        '10': test_mutational_testing,
        '11': test_coverage_enhancer,
        '12': test_documentation_generator,
    }
    
    while True:
        print(MENU)
        choice = input(f"{Colors.BOLD}Enter choice: {Colors.END}").strip().lower()
        
        if choice == 'q':
            print_info("Goodbye! 👋")
            break
        elif choice == 's':
            state.show_status()
        elif choice == 'c':
            state = AgentState()
            print_success("State cleared!")
        elif choice == 'b':
            run_basic_pipeline(state)
        elif choice == 'f':
            run_full_pipeline(state)
        elif choice in agents_map:
            try:
                agents_map[choice](state)
            except Exception as e:
                print_error(f"Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print_error("Invalid choice!")
        
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")


if __name__ == "__main__":
    main()

