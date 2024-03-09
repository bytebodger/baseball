import pg from 'pg';
import { postgresConnection } from '../../constants/postgresConnection.js';

export const getLatestWebSchedule = async () => {
    const { Client } = pg;
    const client = new Client(postgresConnection);
    await client.connect();
    const result = await client.query(`
      SELECT
         web_schedule.web_schedule_id
         ,web_schedule.url
         ,web_schedule.season
      FROM 
         db.web_schedule
      ORDER BY
         web_schedule.season DESC
      LIMIT 
         1
   `);
    await client.end();
    return result;
};
