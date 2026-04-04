import fs from 'node:fs';
import path from 'node:path';

function compareMessagesByTs(left, right) {
    const leftTs = Date.parse(left.ts ?? '');
    const rightTs = Date.parse(right.ts ?? '');

    if (!Number.isNaN(leftTs) && !Number.isNaN(rightTs) && leftTs !== rightTs) {
        return leftTs - rightTs;
    }

    return (left.ts ?? '').localeCompare(right.ts ?? '');
}

export function loadQueueState(queuePath, legacyQueuePath = path.join(path.dirname(queuePath), 'queue.json')) {
    if (fs.existsSync(queuePath)) {
        const raw = fs.readFileSync(queuePath, 'utf8');
        const latestById = new Map();

        for (const line of raw.split(/\r?\n/)) {
            const trimmed = line.trim();
            if (!trimmed) {
                continue;
            }
            const message = JSON.parse(trimmed);
            if (message && typeof message.id === 'string') {
                latestById.set(message.id, message);
            }
        }

        return Array.from(latestById.values()).sort(compareMessagesByTs);
    }

    if (fs.existsSync(legacyQueuePath)) {
        const raw = fs.readFileSync(legacyQueuePath, 'utf8');
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
            return parsed.slice().sort(compareMessagesByTs);
        }
    }

    return [];
}

export function countQueueLogEntries(queuePath) {
    if (!fs.existsSync(queuePath)) {
        return 0;
    }

    return fs.readFileSync(queuePath, 'utf8')
        .split(/\r?\n/)
        .map(line => line.trim())
        .filter(Boolean)
        .length;
}
