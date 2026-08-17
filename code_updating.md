**CODE QUALITY REVIEW CHECKLIST**  
Check if the code has any of these listed issues and try to make code that still passes tests and no longer has these issues.

### 1. Anti-Patterns & Code Smells (Core Design & Maintainability Issues)

**General / Architectural**  
- God-object (a single class/module/file handling too many unrelated concerns; check for excessive public methods/attributes)  
- Big ball of mud (no recognizable structure; tangled dependencies across files)  
- Stovepipe system (isolated components with no integration)  
- Broken modularization / Insufficient modularization / Missing abstraction  
- Multifaceted abstraction / Duplicate abstraction  
- Deficient encapsulation / Unexploited encapsulation  
- Broken hierarchy / Unfactored hierarchy  
- Abstraction inversion  
- Ambiguous viewpoint  
- Interface bloat  
- Race hazard  
- Anemic domain model  
- Call super  
- Circle-ellipse problem  
- Circular dependency  
- Constant interface  
- Object cesspool / Object orgy  
- Sequential coupling  
- Yo-yo problem  
- Action at a distance  
- Boat anchor  
- Busy waiting  
- Caching failure  
- Cargo cult programming  
- **Error hiding** (swallowed exceptions, incomplete error messages, or swallowing expected failures—very common in LLM-generated code)  
- Hard code (literals embedded in source instead of constants/config)  
- Lasagna code (deeply nested conditionals or control flow)  
- Lava flow (code that spreads uncontrollably)  
- Loop-switch sequence  
- Magic strings (string literals used as identifiers instead of enums or constants)  
- Repeating yourself / duplicated code  
- Shooting the messenger (code blaming other components for failures)  
- Soft code (overly flexible but poorly specified logic)  
- Spaghetti code (tangled control flow, unpredictable jumps)  
- Software Peter Principle (documentation and comments that are verbose, misleading, or opposite of intent—e.g., “Claude shits out” experimental journal notes instead of actionable rationale)  
- Too many parameters (long argument lists; indicates poor decomposition)

**Additional Smells** (from standard catalogs)  
- Large class / Long method / Long parameter list  
- Primitive obsession / Data clump  
- Shotgun surgery  
- Refused bequest  
- Downcasting  
- Inner-platform effect  
- Input kludge  
- Magic pushbutton  
- Database-as-IPC  
- Accidental complexity  
- Coding by exception  
- Golden hammer / Reinventing the square wheel / Silver bullet  
- Premature optimization  
- Tester-driven development  
- Dependency hell  
- DLL hell  
- Extension conflict  
- JAR hell  
- Programming by permutation / Programming by accident  

### 2. Linter & Static Analysis Items

**Tooling & Enforcement**  
- Project should pass inspection with: cpplint, clang-format, clang-tidy, pylint, pycodestyle (pep8), pydocstyle, mypy, sonarqube, doxygen, sphinx, or similar.  

### 3. Comments & Documentation

- Experimental results should be in separate report in an organized 
research directory and not the code comments
- No “cargo-cult” comments that restate obvious code.