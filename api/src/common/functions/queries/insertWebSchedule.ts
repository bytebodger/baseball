import { Table } from '../../enums/Table.js';
import { insertIntoTable } from './insertIntoTable.js';

interface Fields {
   has_been_played: boolean,
   html: string,
   season: number,
   time_processed: number | null,
   time_retrieved: number,
   url: string,
}

export const insertWebSchedule = async (fields: Fields) => {
   return await insertIntoTable(Table.webSchedule, fields);
}