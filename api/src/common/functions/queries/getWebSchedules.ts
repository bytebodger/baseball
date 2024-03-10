import pg from 'pg';
import { postgresConnection } from '../../constants/postgresConnection.js';

export const getWebSchedules = async () => {
   const { Client } = pg;
   const client = new Client(postgresConnection);
   await client.connect();
   const result = await client.query(`
      SELECT
         web_schedule.web_schedule_id
         ,web_schedule.url
         ,web_schedule.season
      FROM 
         web_schedule
      ORDER BY
         web_schedule.season DESC
   `)
   await client.end();
   return result;
}