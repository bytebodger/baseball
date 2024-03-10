import dayjs from 'dayjs';
import { getLatestWebSchedule } from './queries/getLatestWebSchedule.js';

export const scrape = () => {
    (async () => {
        const currentYear = dayjs().utc().year();
        const { rowCount, rows: webSchedule } = await getLatestWebSchedule();
        const nextSeason = rowCount ? webSchedule[0].season + 1 : process.env.FIRST_YEAR;
    })();
};
