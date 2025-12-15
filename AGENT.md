# 📘 AGENT.md - PV Uncertainty Tool Master Guide

**Project**: Solar PV Measurement Uncertainty Tool - Professional Edition  
**Repository**: `solar-pv-uncertainty-tool`  
**Live URL**: https://solar-pv-uncertainty-tool.streamlit.app/  
**Last Updated**: December 15, 2024, 10:00 AM IST  
**Version**: 3.0.0-production

---

## 🎯 SACRED PRINCIPLES - Never Compromise

### 1. **UNDERSTAND BEFORE CODING**
```
❌ Jump into coding without context
✅ Read existing code, ask questions, map all dependencies
✅ Understand WHY before implementing HOW
✅ Verify WHICH repository before ANY action
```

### 2. **LOCAL FIRST - Never Web Editor for Code**
```
❌ NEVER manually edit Python in GitHub web editor (tabs/spaces chaos)
❌ NEVER assume indentation will work
✅ ALWAYS use local IDE (Claude Code, VS Code)
✅ ALWAYS validate syntax locally before pushing
✅ Web editor OK ONLY for: README.md, AGENT.md, documentation
```

### 3. **SYNTAX VALIDATION MANDATORY**
```python
# Before EVERY commit:
python -m py_compile filename.py  # Check syntax
flake8 filename.py                 # Check style  
black filename.py --check          # Check formatting
pytest tests/                      # Run tests
```

### 4. **TEST WITH ACTUAL DATA**
```
❌ Assume it works
❌ Test with mock data only
✅ QA test with ACTUAL data entry
✅ Test each page manually
✅ Test complete user workflows
✅ Verify data persists correctly
✅ Check error messages are helpful
```

### 5. **ROLLBACK READY**
```
✅ ALWAYS know how to undo instantly
✅ One fix = one commit (surgical, traceable)
✅ Never batch unrelated changes
✅ Test rollback before deploying
```

### 6. **VERIFY AT EVERY LAYER**
```
Code → Syntax → Local Test → Commit → Deploy → Frontend → Database → User Flow

Each layer MUST pass validation before moving to next
```

---

## 🚨 CRITICAL LESSONS - Never Repeat These Mistakes

### **Lesson 1: Repository Confusion Disaster**
```
❌ WRONG REPO: Worked on `solar-pv-data-analysis` (Railway)
✅ CORRECT REPO: `solar-pv-uncertainty-tool` (Streamlit)
```

**Prevention Protocol:**
1. ✅ ALWAYS check repository dropdown FIRST
2. ✅ Take screenshot to verify correct repo
3. ✅ State repository name explicitly at session start
4. ✅ Cross-reference with live URL
5. ✅ NEVER assume - always confirm visually

### **Lesson 2: Database Schema Mismatch Hell**
```
Problem: Frontend uses `measurement_date` but DB has `created_at`
Cause: models.py changed without SQL migration
Impact: App crashes, data loss, time/money/inventory waste
```

**Prevention Protocol:**
1. ✅ When adding column to models.py → CREATE SQL migration FIRST
2. ✅ Naming: `00X_description_UP.sql` + `00X_description_DOWN.sql`
3. ✅ Test migration UP then DOWN locally
4. ✅ Deploy via Admin page one-click migration runner
5. ✅ Verify frontend-backend nomenclature 100% match
6. ✅ Check: Does frontend code reference this column?
7. ✅ Check: Does database table have this column?
8. ✅ If mismatch → ALTER TABLE or UPDATE frontend code

### **Lesson 3: SQLAlchemy 2.0 Syntax**
```python
# ❌ OLD (breaks in 2.0):
session.execute("SELECT * FROM table")

# ✅ NEW (required):
from sqlalchemy import text
session.execute(text("SELECT * FROM table"))
```

**Always wrap raw SQL in `text()` wrapper!**

### **Lesson 4: Indentation Mixing Catastrophe**
```
❌ Edit Python in web editor → tabs/spaces mix → SyntaxError
✅ Use Claude Code / VS Code → consistent spacing
✅ Validate before push: python -m py_compile file.py
```

---

## 🏗️ Architecture & Development Workflow

### **Question-Driven Development**
Before touching ANY code, ask:

1. **Why is this required?**
2. **Done by whom/which tool?** (Claude Code vs Database ALTER)
3. **Does it need database changes?**
4. **Will it break existing code?**
5. **Is nomenclature consistent across ALL layers?**

### **Phase-Wise Approach**
```
(1) Repository Setup
    ├─ Clear README with architecture
    ├─ Proper .gitignore
    └─ AGENT.md (this file)

(2) Modular Structure
    ├─ database/         (models, connection, migrations)
    ├─ pages/            (Streamlit pages - isolated)
    ├─ utils/            (Helper functions)
    └─ tests/            (Unit & integration tests)

(3) Development (Claude Code IDE)
    ├─ One feature per session
    ├─ Validate syntax locally
    ├─ Test thoroughly
    └─ Commit after working feature

(4) QA Testing
    ├─ Verify line-by-line logic
    ├─ Test data flows
    ├─ Actual data entry
    └─ Error handling

(5) Deployment
    ├─ Push to GitHub
    ├─ Auto-deploys to Streamlit
    ├─ Monitor logs
    └─ Smoke test all pages
```

---

## 🗄️ Database Migration Protocol

### **When to Create Migration**
✅ Adding new table  
✅ Adding new column  
✅ Changing column type  
✅ Adding/removing constraints  
✅ Creating/dropping indexes

### **Migration Template**

**UP Migration** (`migrations/004_add_column_UP.sql`):
```sql
-- Purpose: Add measurement_date column
-- Date: 2024-12-15
-- Author: Gowri

BEGIN;

ALTER TABLE measurements 
ADD COLUMN IF NOT EXISTS measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_measurement_date 
ON measurements(measurement_date);

COMMIT;
```

**DOWN Migration** (`migrations/004_add_column_DOWN.sql`):
```sql
BEGIN;

DROP INDEX IF EXISTS idx_measurement_date;
ALTER TABLE measurements DROP COLUMN IF EXISTS measurement_date;

COMMIT;
```

### **Deployment Process**
1. Create UP and DOWN SQL files
2. Test locally: `psql < migrations/004_add_column_UP.sql`
3. Test rollback: `psql < migrations/004_add_column_DOWN.sql`
4. Test UP again to confirm
5. Push to GitHub
6. In Admin page: Click "Run Migrations" button
7. Verify in app that column exists and works

---

## 🔄 Rollback Procedures

### **Code Rollback**
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)  
git reset --hard HEAD~1

# Revert specific commit
git revert <commit_hash>
```

### **Database Rollback**
```bash
# Via Admin page:
# Click "Rollback Last Migration"

# Manual:
psql $DATABASE_URL < migrations/00X_DOWN.sql
```

### **Streamlit Rollback**
1. Go to Streamlit Cloud dashboard
2. Click app → Settings → Manage app
3. Choose "Reboot app" or deploy previous commit
4. Verify functionality

---

## 📋 Pre-Deployment Checklist

```
[ ] ✅ Correct repository verified (screenshot taken)
[ ] ✅ Python syntax validated locally
[ ] ✅ SQL migrations created (UP & DOWN)
[ ] ✅ Migrations tested locally  
[ ] ✅ Frontend-backend column names match
[ ] ✅ QA tested with ACTUAL data entry
[ ] ✅ All pages load without errors
[ ] ✅ Complete user workflow tested
[ ] ✅ Error handling verified
[ ] ✅ Rollback procedure ready
[ ] ✅ Commit message descriptive
[ ] ✅ No debug/test code left
```

---

## 🎓 Gowri's Vision

**"Innovative, out-of-box thinkers integrated with powerful AI agents making IMPOSSIBLE → I'M POSSIBLE"**

### Core Values:
- ⚡ **Speed with Quality**: Deadline pressure ≠ cutting corners
- 🎯 **Laser Sharp Focus**: Clear thinking even under rush
- 🔬 **Granular Understanding**: Details matter at highest level  
- 💎 **Resource Respect**: No wasted time/money/inventory on rework
- 🤝 **Full Commitment**: Dedicated, systematic error elimination
- 🏆 **Positive Finish**: Success before weekend, not excuses

### Complexity Management:
```
Complex App = Database + Backend + Frontend + Deployment

At each step ask:
✓ Why required?
✓ Which tool? (Claude Code for code, SQL for database)
✓ Database ALTER needed?
✓ Will it break existing code?
✓ Nomenclature consistent?
✓ Can be rolled back?
```

---

## 🔍 Quick Commands

### Verify Repository
```bash
git remote -v
# Should show: ganeshgowri-ASA/solar-pv-uncertainty-tool
```

### Run Locally
```bash
streamlit run streamlit_app.py
```

### Test Database
```python
from database.connection import get_db_session
from sqlalchemy import text

with get_db_session() as session:
    result = session.execute(text("SELECT version();"))
    print(result.fetchone())
```

### Deploy
```bash
git add .
git commit -m "fix: describe the fix"
git push origin main
# Auto-deploys to Streamlit
```

---

## ✅ Success Criteria

```
✅ App loads without errors
✅ All pages functional  
✅ Database operations work
✅ Data persists correctly
✅ Frontend shows accurate data
✅ Helpful error messages
✅ Complete workflow executable
✅ No console errors
✅ Rollback tested
✅ Gowri approves! 🎉
```

---

## 📚 Resources

- **IEC 60891**: Temperature/irradiance correction
- **IEC 61853**: PV module performance testing
- **JCGM 100:2008**: GUM uncertainty guidelines
- **SQLAlchemy 2.0**: https://docs.sqlalchemy.org/en/20/
- **Streamlit**: https://docs.streamlit.io/

---

**Remember**: 
- 🔒 Safe, tested, reversible changes
- 🎯 Quality over speed  
- 🔬 Systematic error elimination
- 🚀 Making IMPOSSIBLE → I'M POSSIBLE!

---

*Last verified: December 15, 2024, 10:00 AM IST*  
*Next review: After each major feature or issue resolution*
