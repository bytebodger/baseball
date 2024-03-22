import dayjs from 'dayjs';
import dayOfYear from 'dayjs/plugin/dayOfYear.js';
import isSameOrAfter from 'dayjs/plugin/isSameOrAfter.js';
import isSameOrBefore from 'dayjs/plugin/isSameOrBefore.js';
import utc from 'dayjs/plugin/utc.js';
import { dbClient } from '../../constants/dbClient.js';
import { retrieveWebSchedules } from '../retrieveWebSchedules.js';

await dbClient.connect();
dayjs.extend(dayOfYear);
dayjs.extend(utc);
dayjs.extend(isSameOrAfter);
dayjs.extend(isSameOrBefore);
await retrieveWebSchedules();
