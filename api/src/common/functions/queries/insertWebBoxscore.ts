import { Table } from '../../enums/Table.js';
import { insertIntoDBTable } from './insertIntoDBTable.js';

interface Fields {
   season: number,
   url: string,
}

export const insertWebBoxscore = async (fields: Fields) => {
   return await insertIntoDBTable(Table.webBoxScore, fields);
}