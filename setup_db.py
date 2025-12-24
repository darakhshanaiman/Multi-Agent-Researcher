import sqlite3

conn = sqlite3.connect('history.db')
c = conn.cursor()

print("🔨 Building Correct Database Schema...")

# 1. USERS Table
c.execute('''CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT,
    "createdAt" TEXT
)''')

# 2. THREADS Table
c.execute('''CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY("userId") REFERENCES users("id")
)''')

# 3. STEPS Table (Fixed: Added 'tags')
c.execute('''CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "streaming" INTEGER NOT NULL,
    "input" TEXT,
    "output" TEXT,
    "isError" INTEGER NOT NULL,
    "createdAt" TEXT,
    "start" TEXT,
    "end" TEXT,
    "defaultOpen" INTEGER,
    "showInput" INTEGER,
    "metadata" TEXT,
    "generation" TEXT,
    "waitForAnswer" INTEGER,
    "language" TEXT,
    "tags" TEXT,  -- <--- NEW COLUMN ADDED
    FOREIGN KEY("threadId") REFERENCES threads("id")
)''')

# 4. ELEMENTS Table (Fixed: Added 'props')
c.execute('''CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT, -- <--- NEW COLUMN ADDED
    FOREIGN KEY("threadId") REFERENCES threads("id")
)''')

# 5. FEEDBACKS Table
c.execute('''CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value" INTEGER NOT NULL,
    "comment" TEXT,
    "strategy" TEXT,
    FOREIGN KEY("threadId") REFERENCES threads("id")
)''')

conn.commit()
conn.close()
print("✅ Database created with correct columns (tags & props)!")